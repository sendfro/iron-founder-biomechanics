import os
import tempfile
import time
import requests 
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from fpdf import FPDF
import sqlite3
from datetime import datetime

from ultralytics import YOLO
import chromadb
from chromadb.utils import embedding_functions
from google import genai

APP_VERSION = "v13.0 (Dual-Engine Fatigue & Comparison Mode)"

# ==================================================
# 1. LOCAL DATABASE SETUP (SQLITE)
# ==================================================
def init_db():
    conn = sqlite3.connect('iron_founder_clients.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  athlete_name TEXT,
                  age TEXT,
                  occupation TEXT,
                  sport TEXT,
                  movement TEXT,
                  primary_metric REAL,
                  secondary_metric REAL,
                  prescription TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ==================================================
# 2. GENERATIVE AI & RAG SETUP
# ==================================================
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

try: ai_client = genai.Client(api_key=GEMINI_API_KEY)
except: ai_client = None

try:
    db_path = os.path.join(os.getcwd(), "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name="pt_recovery_protocols", embedding_function=ef)
except: collection = None

def generate_ai_coaching_plan(movement_telemetry, is_comparison=False):
    if not ai_client or not collection: return "⚠️ API Key or ChromaDB missing."
    try:
        db_results = collection.query(query_texts=[movement_telemetry], n_results=1)
        matched_protocol = db_results['documents'][0][0]
        
        if is_comparison:
            prompt = f"""You are an elite Sports Physical Therapist evaluating fatigue.
            Compare these two sets of telemetry. Diagnose the biomechanical breakdown caused by fatigue/load, and prescribe a specific intervention using the reference.
            DATA: {movement_telemetry}
            REFERENCE: {matched_protocol}
            """
        else:
            prompt = f"""You are an elite Sports Physical Therapist. 
            Synthesize this athlete's telemetry and the clinical reference into a 3-step actionable recovery plan.
            TELEMETRY: {movement_telemetry}
            REFERENCE: {matched_protocol}
            """
        response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e: return f"Error building coaching routine: {e}"

# ==================================================
# 3. APP SETUP & SIDEBAR
# ==================================================
st.set_page_config(page_title="Iron Founder Biomechanics", layout="wide")

with st.sidebar:
    st.header("🗄️ Local Client Database")
    conn = sqlite3.connect('iron_founder_clients.db')
    df = pd.read_sql_query("SELECT date, athlete_name, sport, movement FROM history ORDER BY id DESC", conn)
    conn.close()
    if df.empty: st.info("Database is empty.")
    else: st.dataframe(df, hide_index=True)

st.title("Iron Founder AI: Motion Capture Engine")
st.caption(f"Build: {APP_VERSION}")

# ==================================================
# 4. KINEMATIC MATH & YOLO ENGINE
# ==================================================
def calculate_2d_angle(a, b, c):
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def calculate_trunk_lean(ls, rs, lh, rh):
    mid_s = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
    mid_h = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
    return abs(np.degrees(np.arctan2(mid_s[0] - mid_h[0], mid_h[1] - mid_s[1])))

def detect_valgus(lk, rk, la, ra):
    if np.linalg.norm(ra - la) == 0: return False
    return np.linalg.norm(rk - lk) < np.linalg.norm(ra - la) * 0.8

def analyze_video(uploaded_file, movement_type, ui_container):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as tfile:
        tfile.write(uploaded_file.read()); video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try: model = YOLO('yolov8n-pose.pt') 
    except: return None

    preview = ui_container.empty(); progress_bar = ui_container.progress(0)
    current_frame, valgus_errors = 0, 0
    metric_a_hist, metric_b_hist = [], []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            current_frame += 1
            if current_frame % 2 != 0: continue
            progress_bar.progress(min(current_frame / total_frames, 1.0))

            results = model(cv2.resize(frame, (640, 480)), verbose=False)
            preview.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kps = results[0].keypoints.xy[0].cpu().numpy()
                try:
                    if "Squat" in movement_type:
                        kf = 180 - ((calculate_2d_angle(kps[11], kps[13], kps[15]) + calculate_2d_angle(kps[12], kps[14], kps[16])) / 2)
                        metric_a_hist.append(kf); metric_b_hist.append(calculate_trunk_lean(kps[5], kps[6], kps[11], kps[12]))
                        if detect_valgus(kps[13], kps[14], kps[15], kps[16]): valgus_errors += 1
                    elif "Press" in movement_type:
                        l_el = calculate_2d_angle(kps[5], kps[7], kps[9]); r_el = calculate_2d_angle(kps[6], kps[8], kps[10])
                        metric_a_hist.append((l_el + r_el) / 2); metric_b_hist.append(abs(l_el - r_el))
                    elif "Deadlift" in movement_type:
                        l_h = calculate_2d_angle(kps[5], kps[11], kps[13]); r_h = calculate_2d_angle(kps[6], kps[12], kps[14])
                        metric_a_hist.append((l_h + r_h) / 2); metric_b_hist.append(calculate_trunk_lean(kps[5], kps[6], kps[11], kps[12]))
                except: pass
    finally: cap.release()
    preview.empty(); progress_bar.empty()

    valid_frames = max(len(metric_a_hist), 1)
    report_data = {"hist_a": metric_a_hist, "hist_b": metric_b_hist, "type": movement_type.split(": ")[-1]}
    
    if "Squat" in movement_type: report_data.update({"p": max(metric_a_hist, default=0), "s": max(metric_b_hist, default=0), "r": (valgus_errors/valid_frames)*100})
    elif "Press" in movement_type: report_data.update({"p": max(metric_a_hist, default=0), "s": max(metric_b_hist, default=0), "r": 0})
    else: report_data.update({"p": min(metric_a_hist, default=0), "s": max(metric_a_hist, default=0), "r": max(metric_b_hist, default=0)})

    return report_data

def format_telemetry(rep):
    if rep['type'] == "Squat": return f"Flexion: {rep['p']:.1f}°. Trunk: {rep['s']:.1f}°. Valgus: {rep['r']:.1f}%."
    if rep['type'] == "Overhead Press": return f"Extension: {rep['p']:.1f}°. Asymmetry: {rep['s']:.1f}°."
    if rep['type'] == "Deadlift (Hinge)": return f"Min Hip: {rep['p']:.1f}°. Lockout: {rep['s']:.1f}°. Trunk Lean: {rep['r']:.1f}°."

# ==================================================
# 5. UI: SINGLE VS COMPARISON MODE
# ==================================================
analysis_mode = st.radio("⚙️ Analysis Mode", ["Single Video (Standard)", "Side-by-Side (Fatigue / Intervention)"], horizontal=True)
selected_movement = st.selectbox("🎯 Select Movement Engine", ["Lower Body: Squat", "Upper Body: Overhead Press", "Total Body: Deadlift (Hinge)"])
st.markdown("---")

if analysis_mode == "Single Video (Standard)":
    uploaded_video = st.file_uploader("Upload Movement Video", type=["mp4", "mov"])
    if uploaded_video:
        if st.button("Analyze Mechanics", type="primary"):
            rep = analyze_video(uploaded_video, selected_movement, st)
            st.success("Analysis Complete!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Metric 1", f"{rep['p']:.1f}"); c2.metric("Metric 2", f"{rep['s']:.1f}"); c3.metric("Metric 3", f"{rep['r']:.1f}")
            st.line_chart(pd.DataFrame({"Primary Angle": rep["hist_a"], "Secondary Angle": rep["hist_b"]}))
            st.info(generate_ai_coaching_plan(format_telemetry(rep)))

else:
    # COMPARISON MODE UI
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Video A (e.g., Set 1 / Fresh)")
        vid_A = st.file_uploader("Upload Video A", type=["mp4", "mov"], key="vidA")
    with colB:
        st.subheader("Video B (e.g., Set 5 / Fatigued)")
        vid_B = st.file_uploader("Upload Video B", type=["mp4", "mov"], key="vidB")

    if vid_A and vid_B:
        if st.button("Run Dual-Analysis Engine", type="primary", use_container_width=True):
            col_res_A, col_res_B = st.columns(2)
            with col_res_A: rep_A = analyze_video(vid_A, selected_movement, col_res_A)
            with col_res_B: rep_B = analyze_video(vid_B, selected_movement, col_res_B)
            
            st.markdown("### 📊 Fatigue Delta Metrics")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Metric 1 Change", f"{rep_B['p']:.1f}", delta=f"{rep_B['p'] - rep_A['p']:.1f}", delta_color="inverse")
            mc2.metric("Metric 2 Change", f"{rep_B['s']:.1f}", delta=f"{rep_B['s'] - rep_A['s']:.1f}", delta_color="inverse")
            mc3.metric("Metric 3 Change", f"{rep_B['r']:.1f}", delta=f"{rep_B['r'] - rep_A['r']:.1f}", delta_color="inverse")

            comp_A, comp_B = st.columns(2)
            comp_A.line_chart(pd.DataFrame({"Video A (Fresh)": rep_A["hist_a"]}))
            comp_B.line_chart(pd.DataFrame({"Video B (Fatigued)": rep_B["hist_a"]}))

            st.markdown("---")
            st.header("🧠 AI Fatigue Diagnosis & Prescription")
            combined_telemetry = f"[VIDEO A/FRESH]: {format_telemetry(rep_A)} | [VIDEO B/FATIGUED]: {format_telemetry(rep_B)}"
            with st.spinner("Synthesizing dual-telemetry against knowledge base..."):
                st.info(generate_ai_coaching_plan(combined_telemetry, is_comparison=True))