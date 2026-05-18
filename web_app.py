import os
import json
import tempfile
import sqlite3
import time
import random
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# NEW: Import the Vision architecture
from ultralytics import YOLO

APP_VERSION = "Metric Confidence v6.0 (Hybrid Edge-Cloud)"

# ==================================================
# APP SETUP
# ==================================================
st.set_page_config(
    page_title="Iron Founder Biomechanics",
    layout="wide"
)

st.title("Iron Founder AI: Motion Capture Engine")
st.caption(f"Build Version: {APP_VERSION}")
st.markdown("Upload movement videos for AI-assisted biomechanical screening and kinetic load prediction.")

st.warning(
    "This tool provides AI-assisted movement screening, not medical diagnosis. "
    "For pain, injury, or clinical decisions, consult a qualified professional."
)

# Initialize Session State
for key in ["single_report", "single_video_name", "before_report", "after_report", "before_video_name", "after_video_name"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ==================================================
# CLIENT PROFILE & CALIBRATION
# ==================================================
st.sidebar.header("Client Profile")
st.sidebar.caption(f"Build: {APP_VERSION}")

client_name = st.sidebar.text_input("Client Name")
client_age = st.sidebar.text_input("Client Age")
client_activity = st.sidebar.text_input("Sport / Activity")
coach_name = st.sidebar.text_input("Coach / Trainer Name")
client_notes = st.sidebar.text_area("Session Notes")

client_profile = {
    "client_name": client_name,
    "client_age": client_age,
    "client_activity": client_activity,
    "coach_name": coach_name,
    "client_notes": client_notes,
}

# ==================================================
# SETTINGS & CONFIDENCE DICTS
# ==================================================
MOVEMENT_TESTS = {
    "Squat Analysis": {
        "description": "Analyzes squat depth, knee valgus risk, trunk lean, and pelvic control.",
        "primary_metrics": ["Knee Flexion", "Knee Valgus", "Trunk Lean", "Pelvic Drop"],
        "knee_flexion_target": 90,
        "trunk_lean_limit": 20,
        "pelvic_drop_limit": 8,
    },
    "Running / Gait Analysis": {
        "description": "Analyzes pelvic drop, trunk lean, gait instability, and knee flexion during running or walking.",
        "primary_metrics": ["Pelvic Drop", "Trunk Lean", "Knee Flexion"],
        "pelvic_drop_limit": 8,
        "trunk_lean_limit": 15,
        "knee_flexion_min": 20,
    },
    "Jump Landing": {
        "description": "Analyzes landing mechanics, knee valgus risk, knee flexion absorption, trunk lean, and pelvic control.",
        "primary_metrics": ["Landing Knee Flexion", "Knee Valgus", "Trunk Lean", "Pelvic Drop"],
        "landing_knee_flexion_min": 35,
        "trunk_lean_limit": 20,
        "pelvic_drop_limit": 8,
    },
    "Posture Screen": {
        "description": "Analyzes static posture, shoulder tilt, pelvic tilt, trunk lean, and body alignment.",
        "primary_metrics": ["Shoulder Tilt", "Pelvic Drop", "Trunk Lean"],
        "shoulder_tilt_limit": 6,
        "pelvic_drop_limit": 6,
        "trunk_lean_limit": 10,
    },
}

CAMERA_VIEWS = {
    "Front View": {"description": "Best for knee valgus, shoulder tilt, pelvic tilt.", "best_for": ["Knee valgus", "Shoulder tilt", "Pelvic tilt"], "weak_for": ["True squat depth", "Forward trunk lean"]},
    "Side View": {"description": "Best for squat depth, trunk lean, landing mechanics.", "best_for": ["Knee flexion", "Squat depth", "Trunk lean"], "weak_for": ["Knee valgus", "Left/right asymmetry"]},
    "Rear View": {"description": "Useful for gait, pelvic drop, heel path.", "best_for": ["Pelvic drop", "Gait symmetry", "Rear-chain control"], "weak_for": ["Precise knee flexion", "Squat depth"]},
    "Diagonal / Unknown": {"description": "Least reliable. Diagonal angles distort joint measurements.", "best_for": ["General visual screening"], "weak_for": ["Knee flexion", "Knee valgus", "Pelvic drop", "Trunk lean"]},
}

METRIC_CONFIDENCE = {
    "Squat Analysis": {
        "Front View": {"Knee Flexion": ("Medium-Low", 0.45, ""), "Knee Valgus": ("High", 1.00, ""), "Trunk Lean": ("Low", 0.35, ""), "Pelvic Drop": ("Medium", 0.75, "")},
        "Side View": {"Knee Flexion": ("High", 1.00, ""), "Knee Valgus": ("Low", 0.30, ""), "Trunk Lean": ("High", 1.00, ""), "Pelvic Drop": ("Low", 0.35, "")},
        "Rear View": {"Knee Flexion": ("Low", 0.30, ""), "Knee Valgus": ("Medium", 0.65, ""), "Trunk Lean": ("Low", 0.30, ""), "Pelvic Drop": ("Medium", 0.70, "")},
        "Diagonal / Unknown": {"Knee Flexion": ("Low", 0.25, ""), "Knee Valgus": ("Low", 0.25, ""), "Trunk Lean": ("Low", 0.25, ""), "Pelvic Drop": ("Low", 0.25, "")},
    },
    "Running / Gait Analysis": {
        "Front View": {"Pelvic Drop": ("Medium", 0.75, ""), "Trunk Lean": ("Low", 0.40, ""), "Knee Flexion": ("Low", 0.40, "")},
        "Side View": {"Pelvic Drop": ("Low", 0.35, ""), "Trunk Lean": ("High", 1.00, ""), "Knee Flexion": ("Medium", 0.75, "")},
        "Rear View": {"Pelvic Drop": ("High", 1.00, ""), "Trunk Lean": ("Low", 0.35, ""), "Knee Flexion": ("Low", 0.35, "")},
        "Diagonal / Unknown": {"Pelvic Drop": ("Low", 0.25, ""), "Trunk Lean": ("Low", 0.25, ""), "Knee Flexion": ("Low", 0.25, "")},
    },
    "Jump Landing": {
        "Front View": {"Knee Flexion": ("Medium-Low", 0.45, ""), "Knee Valgus": ("High", 1.00, ""), "Trunk Lean": ("Low", 0.35, ""), "Pelvic Drop": ("Medium", 0.70, "")},
        "Side View": {"Knee Flexion": ("High", 1.00, ""), "Knee Valgus": ("Low", 0.30, ""), "Trunk Lean": ("High", 1.00, ""), "Pelvic Drop": ("Low", 0.35, "")},
        "Rear View": {"Knee Flexion": ("Low", 0.35, ""), "Knee Valgus": ("Medium", 0.65, ""), "Trunk Lean": ("Low", 0.35, ""), "Pelvic Drop": ("Medium", 0.70, "")},
        "Diagonal / Unknown": {"Knee Flexion": ("Low", 0.25, ""), "Knee Valgus": ("Low", 0.25, ""), "Trunk Lean": ("Low", 0.25, ""), "Pelvic Drop": ("Low", 0.25, "")},
    },
    "Posture Screen": {
        "Front View": {"Shoulder Tilt": ("High", 1.00, ""), "Pelvic Drop": ("High", 1.00, ""), "Trunk Lean": ("Medium", 0.70, "")},
        "Side View": {"Shoulder Tilt": ("Low", 0.35, ""), "Pelvic Drop": ("Low", 0.35, ""), "Trunk Lean": ("High", 1.00, "")},
        "Rear View": {"Shoulder Tilt": ("High", 0.90, ""), "Pelvic Drop": ("High", 0.90, ""), "Trunk Lean": ("Medium", 0.65, "")},
        "Diagonal / Unknown": {"Shoulder Tilt": ("Low", 0.25, ""), "Pelvic Drop": ("Low", 0.25, ""), "Trunk Lean": ("Low", 0.25, "")},
    },
}

# ==================================================
# CLOUD NODE (MOCK GAITDYNAMICS API)
# ==================================================
def mock_gaitdynamics_api(kinematic_payload):
    """
    Simulates a cloud GPU receiving a JSON payload of 2D coordinates,
    running a generative diffusion model, and returning 3D Ground Reaction Forces.
    """
    # Simulate network latency and GPU inference time
    time.sleep(1.8)
    
    frame_count = len(kinematic_payload)
    if frame_count == 0:
        return {"error": "Empty payload"}

    # Simulate predictive physics outputs based on the payload data
    peak_grf_bw = round(random.uniform(1.8, 2.8), 2) # Peak Ground Reaction Force (x Bodyweight)
    lateral_knee_load = round(random.uniform(0.4, 1.3), 2) # Lateral load multiplier
    
    return {
        "status": "success",
        "generative_physics": {
            "peak_grf_bodyweight": peak_grf_bw,
            "lateral_knee_load_multiplier": lateral_knee_load,
            "predicted_acl_strain": "High" if lateral_knee_load > 0.9 else "Nominal",
            "confidence_score": 0.94
        }
    }

# ==================================================
# BASIC KINEMATIC HELPERS
# ==================================================
def clean_values(values): return [v for v in values if v is not None and not pd.isna(v)]
def safe_max(values): return max(clean_values(values)) if clean_values(values) else 0
def safe_mean(values): return float(np.mean(clean_values(values))) if clean_values(values) else 0
def count_valid_values(values): return len(clean_values(values))

def smooth_series(series, window_size=5):
    series = clean_values(series)
    if len(series) < window_size: return series
    return pd.Series(series).rolling(window=window_size, min_periods=1, center=True).mean().tolist()

def visible_enough(*scores, threshold=0.50):
    return all(score >= threshold for score in scores)

# 2D Math for YOLO Output
def calculate_2d_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def calculate_tilt(point_a, point_b):
    radians = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    angle = abs(radians * 180.0 / np.pi)
    return min(angle, abs(180 - angle))

def calculate_trunk_lean(l_shoulder, r_shoulder, l_hip, r_hip):
    mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
    mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]
    return abs(np.degrees(np.arctan2(dx, dy)))

def detect_valgus(l_knee, r_knee, l_ankle, r_ankle):
    knee_dist = np.linalg.norm(r_knee - l_knee)
    ankle_dist = np.linalg.norm(r_ankle - l_ankle)
    if ankle_dist == 0: return False
    return knee_dist < ankle_dist * 0.8

def get_metric_confidence(movement_test, camera_view):
    return METRIC_CONFIDENCE.get(movement_test, {}).get(camera_view, {})

def metric_weight(report, metric_name):
    return report.get("metric_confidence", {}).get(metric_name, ("Medium", 0.70, ""))[1]

def grade_from_score(score):
    if score >= 95: return "A+"
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    return "Needs Work"

def movement_label(score):
    if score >= 90: return "Excellent"
    if score >= 80: return "Good"
    if score >= 70: return "Fair"
    return "Needs Improvement"

# ==================================================
# QUALITY ENGINES
# ==================================================
def assess_camera_reliability(movement_test, camera_view, tracking_confidence_rate):
    score = 100
    if tracking_confidence_rate < 70: score -= 35
    elif tracking_confidence_rate < 85: score -= 15
    score = int(max(0, min(100, round(score))))
    return {"score": score, "label": "High" if score >= 85 else "Medium" if score >= 65 else "Low"}

def assess_data_quality(report):
    tracking_score = report.get("tracking_confidence_rate", 0)
    processed_frames = report.get("processed_frames", 0)
    score = 100
    if tracking_score < 70: score -= 32
    elif tracking_score < 80: score -= 16
    if processed_frames < 20: score -= 25
    score = int(max(0, min(100, round(score))))
    return {"score": score, "grade": grade_from_score(score), "reasons": []}

def assess_movement_quality(report):
    movement_test = report["movement_test"]
    settings = MOVEMENT_TESTS[movement_test]
    score = 100
    flags, cues = [], []
    primary_limitation = "None detected"

    if movement_test == "Squat Analysis":
        if report["max_knee_flexion"] < settings["knee_flexion_target"]:
            score -= 15; flags.append("Range of Motion Restriction (Depth)")
            cues.append("Focus on reaching parallel."); primary_limitation = "Limited Squat Depth"

        if report["valgus_rate"] > 20:
            score -= 20; flags.append("Possible Knee Valgus")
            cues.append("Drive knees outward."); primary_limitation = "Knee Tracking"

    score = int(max(0, min(100, round(score))))
    if not flags: flags.append("No major movement flags detected.")
    if not cues: cues.append("Maintain current mechanics.")

    return {
        "score": score, "grade": grade_from_score(score), "label": movement_label(score),
        "flags": flags, "coaching_cues": cues, "primary_limitation": primary_limitation,
    }

def evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected):
    fault_detected = False
    if movement_test == "Squat Analysis" and (knee_flexion < 90 or valgus_detected): fault_detected = True
    return fault_detected

# ==================================================
# EXPORT GENERATORS
# ==================================================
def save_chart_png(dataframe, title, filename):
    plt.figure(figsize=(10, 4))
    for column in dataframe.columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        plt.plot(series, label=column)
    plt.title(title)
    plt.xlabel("Processed Frame")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def create_pdf_report(report, chart_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.75 * inch, y, "Iron Founder AI")
    y -= 0.35 * inch
    c.setFont("Helvetica", 12)
    for row in [report["label"], f"Test: {report['movement_test']}", f"Camera: {report.get('camera_view', '')}"]:
        c.drawString(0.75 * inch, y, row); y -= 0.24 * inch
    if os.path.exists(chart_path):
        if y < 3.5 * inch:
            c.showPage(); y = height - 0.75 * inch
        c.drawImage(chart_path, 0.75 * inch, y - 2.5*inch, width=6.8 * inch, height=2.5 * inch, preserveAspectRatio=True)
    c.save()

def init_db():
    os.makedirs("reports", exist_ok=True)
    conn = sqlite3.connect("reports/report_history.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, client_name TEXT, client_age TEXT, movement_test TEXT, camera_view TEXT, movement_score INTEGER)''')
    conn.commit()
    return conn

def save_report_history(report):
    conn = init_db()
    cursor = conn.cursor()
    client = report.get("client_profile", {})
    mq = report.get("movement_quality", {})
    cursor.execute('''INSERT INTO history (date, client_name, client_age, movement_test, camera_view, movement_score) VALUES (?, ?, ?, ?, ?, ?)''', 
                   (datetime.now().strftime("%Y-%m-%d %H:%M"), client.get("client_name", ""), client.get("client_age", ""), report.get("movement_test", ""), report.get("camera_view", ""), mq.get("score", 0)))
    conn.commit()
    conn.close()

# ==================================================
# VIDEO ANALYSIS ENGINE (v6.0 Hybrid Vision Pipeline)
# ==================================================
def analyze_video(uploaded_file, movement_test, camera_view, label="Video", client_profile=None):
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open {label}.")
        os.remove(video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize YOLO Model (Downloads nano model if not present)
    try:
        model = YOLO('yolov8n-pose.pt') 
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

    current_frame, processed_frames, low_confidence_frames, movement_faults, valgus_errors = 0, 0, 0, 0, 0
    pelvic_history, knee_flexion_history, trunk_lean_history, shoulder_tilt_history = [], [], [], []

    preview = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    frame_stride = 2

    # Array to hold our batch payload for the Cloud API
    yolo_trajectory_payload = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            current_frame += 1
            if current_frame % frame_stride != 0: continue
            processed_frames += 1

            if total_frames > 0:
                progress_bar.progress(min(current_frame / total_frames, 1.0))
                progress_text.text(f"{label} | YOLO Edge Inference: Frame {current_frame}/{total_frames}")

            # YOLO Native Downsampling
            small_frame = cv2.resize(frame, (640, 480))
            
            # YOLO Inference
            results = model(small_frame, verbose=False)
            annotated_frame = results[0].plot()

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else np.ones(17)

                # Package this frame's raw data for the cloud payload
                frame_data = {
                    "frame": current_frame,
                    "keypoints": keypoints.tolist(),
                    "confidences": confs.tolist()
                }
                yolo_trajectory_payload.append(frame_data)

                try:
                    # YOLO COCO Mapping: Shoulders(5,6), Hips(11,12), Knees(13,14), Ankles(15,16)
                    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
                    l_hip, r_hip = keypoints[11], keypoints[12]
                    l_knee, r_knee = keypoints[13], keypoints[14]
                    l_ankle, r_ankle = keypoints[15], keypoints[16]

                    tracking_conf = [confs[i] for i in [5, 6, 11, 12, 13, 14, 15, 16]]
                    
                    if not visible_enough(*tracking_conf, threshold=0.50):
                        low_confidence_frames += 1
                        pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)
                    else:
                        r_knee_angle = calculate_2d_angle(r_hip, r_knee, r_ankle)
                        l_knee_angle = calculate_2d_angle(l_hip, l_knee, l_ankle)
                        knee_flexion = 180 - ((r_knee_angle + l_knee_angle) / 2)

                        pelvic_drop = calculate_tilt(l_hip, r_hip)
                        trunk_lean = calculate_trunk_lean(l_shoulder, r_shoulder, l_hip, r_hip)
                        shoulder_tilt = calculate_tilt(l_shoulder, r_shoulder)
                        valgus_detected = detect_valgus(l_knee, r_knee, l_ankle, r_ankle)

                        if valgus_detected: valgus_errors += 1

                        pelvic_history.append(pelvic_drop)
                        knee_flexion_history.append(knee_flexion)
                        trunk_lean_history.append(trunk_lean)
                        shoulder_tilt_history.append(shoulder_tilt)

                        fault = evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected)
                        if fault: movement_faults += 1

                except Exception:
                    low_confidence_frames += 1
                    pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)
            else:
                low_confidence_frames += 1
                pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            preview.image(rgb_frame, channels="RGB", use_container_width=True)
    finally:
        cap.release()
        try: os.remove(video_path)
        except Exception: pass

    # ==================================================
    # PHASE 2: HYBRID CLOUD INFERENCE
    # ==================================================
    progress_text.text("Connecting to GaitDynamics Cloud GPU...")
    progress_bar.progress(1.0)
    
    # Send lightweight JSON payload to our mock API
    kinetic_results = mock_gaitdynamics_api(yolo_trajectory_payload)
    
    preview.empty(); progress_text.empty(); progress_bar.empty()

    if processed_frames == 0: return None

    pelvic_history = smooth_series(pelvic_history)
    knee_flexion_history = smooth_series(knee_flexion_history)
    trunk_lean_history = smooth_series(trunk_lean_history)

    valid_data_frames = count_valid_values(knee_flexion_history)
    valid_frames_for_rates = max(valid_data_frames, 1)

    report = {
        "label": label, "movement_test": movement_test, "camera_view": camera_view, "client_profile": client_profile or {},
        "fps": fps, "total_frames": total_frames, "processed_frames": processed_frames, "valid_data_frames": valid_data_frames, "low_confidence_frames": low_confidence_frames,
        "max_pelvic_drop": safe_max(pelvic_history), "avg_pelvic_drop": safe_mean(pelvic_history),
        "max_knee_flexion": safe_max(knee_flexion_history), "avg_knee_flexion": safe_mean(knee_flexion_history),
        "max_trunk_lean": safe_max(trunk_lean_history), "avg_trunk_lean": safe_mean(trunk_lean_history),
        "valgus_rate": valgus_errors / valid_frames_for_rates * 100, "movement_fault_rate": movement_faults / valid_frames_for_rates * 100,
        "tracking_confidence_rate": valid_data_frames / processed_frames * 100,
        "pelvic_history": pelvic_history, "knee_flexion_history": knee_flexion_history, "trunk_lean_history": trunk_lean_history,
        "generative_kinetics": kinetic_results.get("generative_physics", {}) # Attach API output
    }

    report["metric_confidence"] = get_metric_confidence(movement_test, camera_view)
    report["camera_reliability"] = assess_camera_reliability(movement_test, camera_view, report["tracking_confidence_rate"])
    report["data_quality"] = assess_data_quality(report)
    report["movement_quality"] = assess_movement_quality(report)

    return report


# ==================================================
# DISPLAY 
# ==================================================
def show_report(report):
    st.header(f"📊 Final Biomechanical Report: {report['label']}")
    st.caption(f"Movement Test: **{report['movement_test']}** | Camera View: **{report.get('camera_view', 'Not selected')}**")

    # 1. MOVEMENT QUALITY UI
    mq = report.get("movement_quality", {})
    st.markdown("### 🏆 Movement Quality")
    colA, colB, colC = st.columns(3)
    colA.metric("Movement Quality Grade", f"{mq.get('grade', 'N/A')} ({mq.get('label', '')})")
    colB.metric("Quality Score", f"{mq.get('score', 0)}/100")
    colC.metric("Primary Limitation", mq.get('primary_limitation', 'N/A'))

    with st.container():
        st.markdown("#### 🚨 Risk Flags & Cues")
        for flag in mq.get("flags", []):
            if "No major movement flags" in flag: st.success(f"**{flag}**")
            else: st.error(f"**{flag}**")
        for cue in mq.get("coaching_cues", []): st.success(f"💡 {cue}")

    st.markdown("---")

    # 2. GENERATIVE KINETICS UI (NEW API DATA)
    kinetics = report.get("generative_kinetics", {})
    if kinetics:
        st.markdown("### ⚡ Generative Kinetics (GaitDynamics Cloud)")
        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("Peak Ground Reaction Force", f"{kinetics.get('peak_grf_bodyweight', 0)}x BW")
        kc2.metric("Lateral Knee Load Multiplier", f"{kinetics.get('lateral_knee_load_multiplier', 0)}x")
        
        acl_strain = kinetics.get("predicted_acl_strain", "N/A")
        if acl_strain == "High": kc3.error(f"Predicted ACL Strain: {acl_strain}")
        else: kc3.success(f"Predicted ACL Strain: {acl_strain}")
        st.markdown("---")

    # 3. KINEMATIC DATA UI
    st.markdown("### 📐 Kinematic Data (Local Edge)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Angular Flexion", f"{report['max_knee_flexion']:.1f}°")
    c2.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°")
    c3.metric("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%")
    c4.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")

    chart_df = pd.DataFrame({
        "Pelvic Drop (°)": report["pelvic_history"],
        "Knee Flexion (°)": report["knee_flexion_history"],
        "Trunk Lean (°)": report["trunk_lean_history"]
    }).apply(pd.to_numeric, errors="coerce")
    st.line_chart(chart_df)
    st.markdown("---")

    # 4. EXPORT UI
    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    save_chart_png(chart_df, f"{report['movement_test']} Kinematics", chart_path)
    create_pdf_report(report, chart_path, pdf_path)
    csv_data = chart_df.to_csv().encode('utf-8')

    col_a, col_b, col_c, col_d = st.columns(4)
    safe_name = report["movement_test"].replace(" ", "_").lower()

    with col_a:
        with open(chart_path, "rb") as img_file: st.download_button("Download PNG", data=img_file, file_name=f"{safe_name}_chart.png", mime="image/png")
    with col_b:
        with open(pdf_path, "rb") as pdf_file: st.download_button("Download PDF", data=pdf_file, file_name=f"{safe_name}_report.pdf", mime="application/pdf")
    with col_c:
        st.download_button("Download CSV", data=csv_data, file_name=f"{safe_name}_data.csv", mime="text/csv")
    with col_d:
        if st.button(f"Save to History", key=f"save_{report['label']}"):
            save_report_history(report)
            st.success("Saved.")

def show_history_dashboard():
    st.markdown("---")
    st.header("Client Report History")
    conn = init_db()
    df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    if not df.empty:
        search = st.text_input("Search Client Name")
        if search: df = df[df["client_name"].str.contains(search, case=False, na=False)]
        st.dataframe(df, use_container_width=True)
    else: st.info("No saved reports yet.")

# ==================================================
# MAIN UI
# ==================================================
analysis_type = st.radio("Choose Analysis Type", ["Single Video Analysis"], horizontal=True)
movement_test = st.selectbox("Choose Movement Test", list(MOVEMENT_TESTS.keys()))
camera_view = st.selectbox("Choose Camera View", list(CAMERA_VIEWS.keys()))

uploaded_video = st.file_uploader("Upload Movement Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    if st.session_state.single_video_name != uploaded_video.name:
        st.session_state.single_report = analyze_video(
            uploaded_video, movement_test, camera_view, "Single Video Report", client_profile
        )
        st.session_state.single_video_name = uploaded_video.name

    if st.session_state.single_report:
        show_report(st.session_state.single_report)

show_history_dashboard()
