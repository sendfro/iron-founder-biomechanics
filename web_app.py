import os
import json
import tempfile
import sqlite3
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

APP_VERSION = "Metric Confidence v4.1 (Engineered)"

# ==================================================
# APP SETUP
# ==================================================
st.set_page_config(
    page_title="Iron Founder Biomechanics",
    layout="wide"
)

st.title("Iron Founder AI: Motion Capture Engine")
st.caption(f"Build Version: {APP_VERSION}")
st.markdown("Upload movement videos for AI-assisted biomechanical screening.")

st.warning(
    "This tool provides AI-assisted movement screening, not medical diagnosis. "
    "For pain, injury, or clinical decisions, consult a qualified professional."
)

# Initialize Session State
if "single_report" not in st.session_state:
    st.session_state.single_report = None
if "single_video_name" not in st.session_state:
    st.session_state.single_video_name = None
if "before_report" not in st.session_state:
    st.session_state.before_report = None
if "after_report" not in st.session_state:
    st.session_state.after_report = None
if "before_video_name" not in st.session_state:
    st.session_state.before_video_name = None
if "after_video_name" not in st.session_state:
    st.session_state.after_video_name = None


# ==================================================
# CLIENT PROFILE
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
# SETTINGS
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
    "Front View": {
        "description": "Best for knee valgus, shoulder tilt, pelvic tilt, and left/right symmetry.",
        "best_for": ["Knee valgus", "Shoulder tilt", "Pelvic tilt", "Left/right asymmetry"],
        "weak_for": ["True squat depth", "Forward trunk lean", "Precise knee flexion"],
    },
    "Side View": {
        "description": "Best for squat depth, knee flexion, hip hinge, trunk lean, and landing mechanics.",
        "best_for": ["Knee flexion", "Squat depth", "Trunk lean", "Landing absorption"],
        "weak_for": ["Knee valgus", "Left/right asymmetry", "Pelvic drop"],
    },
    "Rear View": {
        "description": "Useful for gait, pelvic drop, heel path, and left/right control from behind.",
        "best_for": ["Pelvic drop", "Gait symmetry", "Rear-chain control", "Foot path observation"],
        "weak_for": ["Precise knee flexion", "Squat depth", "Forward trunk lean"],
    },
    "Diagonal / Unknown": {
        "description": "Least reliable. Diagonal angles distort joint measurements and symmetry readings.",
        "best_for": ["General visual screening only"],
        "weak_for": ["Knee flexion", "Knee valgus", "Pelvic drop", "Trunk lean", "Shoulder tilt"],
    },
}

METRIC_CONFIDENCE = {
    "Squat Analysis": {
        "Front View": {"Knee Flexion": ("Medium-Low", 0.45, "Depth estimate from front view is lower confidence."), "Knee Valgus": ("High", 1.00, "Front view is strong for knee valgus."), "Trunk Lean": ("Low", 0.35, "Trunk lean from front view is lower confidence."), "Pelvic Drop": ("Medium", 0.75, "Front view can screen pelvic asymmetry.")},
        "Side View": {"Knee Flexion": ("High", 1.00, "Side view is strong for true squat depth."), "Knee Valgus": ("Low", 0.30, "Side view is weak for knee valgus."), "Trunk Lean": ("High", 1.00, "Side view is strong for trunk lean."), "Pelvic Drop": ("Low", 0.35, "Side view is weak for asymmetry.")},
        "Rear View": {"Knee Flexion": ("Low", 0.30, "Rear view is weak for squat depth."), "Knee Valgus": ("Medium", 0.65, "Rear view can roughly screen lower-body alignment."), "Trunk Lean": ("Low", 0.30, "Rear view is weak for trunk lean."), "Pelvic Drop": ("Medium", 0.70, "Rear view can screen pelvic control.")},
        "Diagonal / Unknown": {"Knee Flexion": ("Low", 0.25, "Diagonal distorts depth."), "Knee Valgus": ("Low", 0.25, "Diagonal distorts knee tracking."), "Trunk Lean": ("Low", 0.25, "Diagonal distorts trunk lean."), "Pelvic Drop": ("Low", 0.25, "Diagonal distorts alignment.")},
    },
    "Running / Gait Analysis": {
        "Front View": {"Pelvic Drop": ("Medium", 0.75, "Front view can screen pelvic control."), "Trunk Lean": ("Low", 0.40, "Front view is weak for trunk lean."), "Knee Flexion": ("Low", 0.40, "Front view is weak for precise knee flexion.")},
        "Side View": {"Pelvic Drop": ("Low", 0.35, "Side view is weak for pelvic drop."), "Trunk Lean": ("High", 1.00, "Side view is strong for running posture."), "Knee Flexion": ("Medium", 0.75, "Side view can screen knee motion.")},
        "Rear View": {"Pelvic Drop": ("High", 1.00, "Rear view is strong for pelvic drop."), "Trunk Lean": ("Low", 0.35, "Rear view is weak for trunk lean."), "Knee Flexion": ("Low", 0.35, "Rear view is weak for precise knee flexion.")},
        "Diagonal / Unknown": {"Pelvic Drop": ("Low", 0.25, "Distorts gait symmetry."), "Trunk Lean": ("Low", 0.25, "Distorts trunk lean."), "Knee Flexion": ("Low", 0.25, "Distorts joint angles.")},
    },
    "Jump Landing": {
        "Front View": {"Knee Flexion": ("Medium-Low", 0.45, "Landing depth is lower confidence."), "Knee Valgus": ("High", 1.00, "Front view is strong for valgus."), "Trunk Lean": ("Low", 0.35, "Trunk lean is lower confidence."), "Pelvic Drop": ("Medium", 0.70, "Front view can screen control.")},
        "Side View": {"Knee Flexion": ("High", 1.00, "Side view is strong for absorption."), "Knee Valgus": ("Low", 0.30, "Side view is weak for valgus."), "Trunk Lean": ("High", 1.00, "Side view is strong for landing trunk control."), "Pelvic Drop": ("Low", 0.35, "Side view is weak for pelvic asymmetry.")},
        "Rear View": {"Knee Flexion": ("Low", 0.35, "Rear view is weak for landing depth."), "Knee Valgus": ("Medium", 0.65, "Rear view can roughly screen tracking."), "Trunk Lean": ("Low", 0.35, "Rear view is weak for trunk lean."), "Pelvic Drop": ("Medium", 0.70, "Rear view can screen control.")},
        "Diagonal / Unknown": {"Knee Flexion": ("Low", 0.25, "Distorts depth."), "Knee Valgus": ("Low", 0.25, "Distorts tracking."), "Trunk Lean": ("Low", 0.25, "Distorts trunk lean."), "Pelvic Drop": ("Low", 0.25, "Distorts symmetry.")},
    },
    "Posture Screen": {
        "Front View": {"Shoulder Tilt": ("High", 1.00, "Front view is strong for shoulder tilt."), "Pelvic Drop": ("High", 1.00, "Front view is strong for pelvic tilt."), "Trunk Lean": ("Medium", 0.70, "Front view can screen trunk lean.")},
        "Side View": {"Shoulder Tilt": ("Low", 0.35, "Side view is weak for shoulder tilt."), "Pelvic Drop": ("Low", 0.35, "Side view is weak for pelvic tilt."), "Trunk Lean": ("High", 1.00, "Side view is strong for trunk alignment.")},
        "Rear View": {"Shoulder Tilt": ("High", 0.90, "Rear view can assess shoulder asymmetry."), "Pelvic Drop": ("High", 0.90, "Rear view can assess pelvic asymmetry."), "Trunk Lean": ("Medium", 0.65, "Rear view can screen trunk lean.")},
        "Diagonal / Unknown": {"Shoulder Tilt": ("Low", 0.25, "Distorts alignment."), "Pelvic Drop": ("Low", 0.25, "Distorts alignment."), "Trunk Lean": ("Low", 0.25, "Distorts alignment.")},
    },
}

# ==================================================
# HELPERS
# ==================================================
def grade_from_score(score):
    if score >= 95: return "A+"
    if score >= 90: return "A"
    if score >= 85: return "B+"
    if score >= 80: return "B"
    if score >= 75: return "C+"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "Needs Work"

def movement_label(score):
    if score >= 90: return "Excellent"
    if score >= 80: return "Good"
    if score >= 70: return "Fair"
    if score >= 60: return "Needs Improvement"
    return "Needs Work"

def clean_values(values):
    return [v for v in values if v is not None and not pd.isna(v)]

def safe_max(values):
    values = clean_values(values)
    return max(values) if values else 0

def safe_mean(values):
    values = clean_values(values)
    return float(np.mean(values)) if values else 0

def count_valid_values(values):
    return len(clean_values(values))

def normalize_time_series(series, target_length=100):
    """Aligns arrays of different lengths for accurate temporal comparison."""
    series = clean_values(series)
    if not series:
        return [0] * target_length
    x_old = np.linspace(0, 1, len(series))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, series).tolist()

def calculate_angle_3d(a, b, c):
    """Calculates angle utilizing the 3D Z-axis from MediaPipe."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))

def get_lm(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return [lm.x, lm.y, lm.z], lm.visibility

def visible_enough(*scores, threshold=0.55):
    return all(score >= threshold for score in scores)

def calculate_line_tilt(point_a, point_b):
    # Kept as 2D since it measures tilt against the camera frame's Y axis
    radians = np.arctan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
    angle = abs(radians * 180.0 / np.pi)
    return min(angle, abs(180 - angle))

def calculate_pelvic_drop(left_hip, right_hip):
    return calculate_line_tilt(left_hip, right_hip)

def calculate_shoulder_tilt(left_shoulder, right_shoulder):
    return calculate_line_tilt(left_shoulder, right_shoulder)

def calculate_trunk_lean(left_shoulder, right_shoulder, left_hip, right_hip):
    mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_hip[1] - mid_shoulder[1]
    return abs(np.degrees(np.arctan2(dx, dy)))

def detect_valgus(left_knee, right_knee, left_ankle, right_ankle):
    knee_dist = np.linalg.norm(np.array(right_knee[:2]) - np.array(left_knee[:2]))
    ankle_dist = np.linalg.norm(np.array(right_ankle[:2]) - np.array(left_ankle[:2]))
    if ankle_dist == 0: return False
    return knee_dist < ankle_dist * 0.8

def save_chart_png(dataframe, title, filename):
    plt.figure(figsize=(10, 4))
    for column in dataframe.columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        plt.plot(series, label=column)
    plt.title(title)
    plt.xlabel("Processed Frame (% Completion)")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def draw_wrapped_text(c, text, x, y, max_chars=90, line_height=0.18 * inch):
    if not text: return y
    words = str(text).split()
    lines, current_line = [], ""
    for word in words:
        possible_line = (current_line + " " + word).strip()
        if len(possible_line) <= max_chars:
            current_line = possible_line
        else:
            if current_line: lines.append(current_line)
            current_line = word
    if current_line: lines.append(current_line)
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height
    return y

def get_metric_confidence(movement_test, camera_view):
    return METRIC_CONFIDENCE.get(movement_test, {}).get(camera_view, {})

def metric_weight(report, metric_name):
    confidence = report.get("metric_confidence", {})
    return confidence.get(metric_name, ("Medium", 0.70, ""))[1]

# ==================================================
# QUALITY ENGINES
# ==================================================
def assess_camera_reliability(movement_test, camera_view, tracking_confidence_rate):
    score = 100
    warnings, strengths = [], []

    if tracking_confidence_rate < 70:
        score -= 35
        warnings.append("Tracking confidence was low. Results should be interpreted carefully.")
    elif tracking_confidence_rate < 85:
        score -= 15
        warnings.append("Tracking confidence was moderate. Results are usable but not ideal.")
    else:
        strengths.append("Tracking confidence was strong.")

    metric_confidence = get_metric_confidence(movement_test, camera_view)
    weak_metrics = [m for m, d in metric_confidence.items() if d[0] in ["Low", "Medium-Low"]]

    if weak_metrics:
        score -= min(20, len(weak_metrics) * 5)
        warnings.append("Lower confidence metrics from this view: " + ", ".join(weak_metrics))

    strong_metrics = [m for m, d in metric_confidence.items() if d[0] == "High"]
    if strong_metrics:
        strengths.append("Strong metrics for this view: " + ", ".join(strong_metrics))

    score = int(max(0, min(100, round(score))))
    label = "High" if score >= 85 else "Medium" if score >= 65 else "Low"

    return {"score": score, "label": label, "warnings": warnings, "strengths": strengths}

def assess_data_quality(report):
    camera_score = report.get("camera_reliability", {}).get("score", 0)
    tracking_score = report.get("tracking_confidence_rate", 0)
    processed_frames = report.get("processed_frames", 0)
    valid_data_frames = report.get("valid_data_frames", 0)

    score = 100
    reasons = []

    if camera_score >= 85: reasons.append("Camera setup was strong.")
    elif camera_score >= 65: score -= 12; reasons.append("Camera setup was usable but not ideal.")
    else: score -= 28; reasons.append("Camera setup reduced report reliability.")

    if tracking_score >= 90: reasons.append("Pose tracking was excellent.")
    elif tracking_score >= 80: score -= 8; reasons.append("Pose tracking was good but not perfect.")
    elif tracking_score >= 70: score -= 16; reasons.append("Pose tracking was moderate.")
    else: score -= 32; reasons.append("Pose tracking was low.")

    if processed_frames < 20: score -= 25; reasons.append("Very few frames were processed.")
    elif processed_frames < 50: score -= 12; reasons.append("Limited frame sample.")
    else: reasons.append("Frame sample size was adequate.")

    valid_ratio = valid_data_frames / max(processed_frames, 1)
    if valid_ratio >= 0.9: reasons.append("Most processed frames contained valid landmarks.")
    elif valid_ratio >= 0.75: score -= 8; reasons.append("Some frames were excluded due to weak landmark confidence.")
    elif valid_ratio >= 0.6: score -= 16; reasons.append("A meaningful number of frames were excluded.")
    else: score -= 30; reasons.append("Too many frames had weak landmark confidence.")

    score = int(max(0, min(100, round(score))))
    return {"score": score, "grade": grade_from_score(score), "reasons": reasons}

def assess_movement_quality(report):
    movement_test = report["movement_test"]
    settings = MOVEMENT_TESTS[movement_test]
    score = 100
    flags, cues, positives, retest_warnings = [], [], [], []
    primary_limitation = "None detected"
    retest_view = "Repeat with the same view if tracking quality was high."

    def apply_penalty(base_penalty, metric_name):
        return base_penalty * metric_weight(report, metric_name)

    if movement_test == "Squat Analysis":
        depth_target = settings["knee_flexion_target"]
        trunk_limit = settings["trunk_lean_limit"]
        pelvic_limit = settings["pelvic_drop_limit"]

        if report["max_knee_flexion"] < depth_target:
            base = min(25, (depth_target - report["max_knee_flexion"]) * 1.5)
            score -= apply_penalty(base, "Knee Flexion")
            flags.append("Range of Motion Restriction (Depth)")
            cues.append("Focus on reaching parallel; elevate heels if ankle mobility is restricted.")
            primary_limitation = "Limited Squat Depth"
            retest_view = "Side View"
        else: positives.append("Squat depth reached the selected target.")

        if report["valgus_rate"] > 20:
            base = min(30, report["valgus_rate"] * 0.6)
            score -= apply_penalty(base, "Knee Valgus")
            flags.append("Possible Knee Valgus")
            cues.append("Drive knees outward and keep tracking over second toes.")
            if primary_limitation == "None detected": primary_limitation = "Knee Tracking"
            retest_view = "Front View"
        elif report["valgus_rate"] > 5:
            base = min(15, report["valgus_rate"] * 0.4)
            score -= apply_penalty(base, "Knee Valgus")
            flags.append("Mild Knee Valgus Tendency")
            cues.append("Control knee position during the lowering and rising phase.")

        if report["max_trunk_lean"] > trunk_limit:
            base = min(25, (report["max_trunk_lean"] - trunk_limit) * 2)
            score -= apply_penalty(base, "Trunk Lean")
            flags.append("Excessive Trunk Lean")
            cues.append("Brace the trunk and keep chest position controlled.")
            if primary_limitation == "None detected": primary_limitation = "Trunk Control"
            retest_view = "Side View"

    elif movement_test == "Running / Gait Analysis":
        if report["max_pelvic_drop"] > settings["pelvic_drop_limit"]:
            score -= apply_penalty(min(35, (report["max_pelvic_drop"] - settings["pelvic_drop_limit"]) * 3), "Pelvic Drop")
            flags.append("Pelvic Drop")
            cues.append("Improve hip stability and keep pelvis level during stance.")
            primary_limitation = "Pelvic Control"
            retest_view = "Rear View"

        if report["max_trunk_lean"] > settings["trunk_lean_limit"]:
            score -= apply_penalty(min(25, (report["max_trunk_lean"] - settings["trunk_lean_limit"]) * 2), "Trunk Lean")
            flags.append("Excessive Trunk Lean")
            cues.append("Run tall with a controlled forward lean from the ankles.")
            if primary_limitation == "None detected": primary_limitation = "Trunk Lean"
            retest_view = "Side View"

    elif movement_test == "Jump Landing":
        if report["max_knee_flexion"] < settings["landing_knee_flexion_min"]:
            score -= apply_penalty(min(35, (settings["landing_knee_flexion_min"] - report["max_knee_flexion"]) * 2), "Knee Flexion")
            flags.append("Stiff Landing")
            cues.append("Land softer by bending hips and knees to absorb force.")
            primary_limitation = "Landing Absorption"
            retest_view = "Side View"

        if report["valgus_rate"] > 20:
            score -= apply_penalty(min(30, report["valgus_rate"] * 0.6), "Knee Valgus")
            flags.append("Landing Knee Valgus")
            cues.append("Land with knees tracking over toes.")
            if primary_limitation == "None detected": primary_limitation = "Landing Knee Valgus"
            retest_view = "Front View"

    elif movement_test == "Posture Screen":
        if report["max_shoulder_tilt"] > settings["shoulder_tilt_limit"]:
            score -= apply_penalty(min(25, (report["max_shoulder_tilt"] - settings["shoulder_tilt_limit"]) * 3), "Shoulder Tilt")
            flags.append("Shoulder Tilt")
            cues.append("Check shoulder height and ribcage position.")
            primary_limitation = "Shoulder Alignment"
            retest_view = "Front View"

        if report["max_pelvic_drop"] > settings["pelvic_drop_limit"]:
            score -= apply_penalty(min(25, (report["max_pelvic_drop"] - settings["pelvic_drop_limit"]) * 3), "Pelvic Drop")
            flags.append("Pelvic Tilt")
            cues.append("Balance weight evenly and retest.")
            if primary_limitation == "None detected": primary_limitation = "Pelvic Alignment"

    for metric, data in report.get("metric_confidence", {}).items():
        if data[0] in ["Low", "Medium-Low"]:
            retest_warnings.append(data[2])

    score = int(max(0, min(100, round(score))))
    if not flags: flags.append("No major movement flags detected.")
    if not cues: cues.append("Maintain current mechanics and retest periodically.")

    return {
        "score": score,
        "grade": grade_from_score(score),
        "label": movement_label(score),
        "flags": flags,
        "coaching_cues": cues[:3],
        "positives": positives,
        "primary_limitation": primary_limitation,
        "recommended_retest_view": retest_view,
        "retest_warnings": list(dict.fromkeys(retest_warnings))[:4],
    }

def evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected):
    settings = MOVEMENT_TESTS[movement_test]
    warning_text = f"{movement_test.upper()}: FORM SOLID"
    warning_color = (0, 255, 0)
    fault_detected = False

    if movement_test == "Squat Analysis":
        warning_text = f"SQUAT | Knee Flexion: {knee_flexion:.1f}°"
        if knee_flexion < settings["knee_flexion_target"]:
            warning_text, warning_color, fault_detected = "SQUAT WARNING: LIMITED DEPTH", (255, 165, 0), True
        if valgus_detected:
            warning_text, warning_color, fault_detected = "SQUAT WARNING: KNEE VALGUS", (255, 0, 0), True
    return warning_text, warning_color, fault_detected

def generate_notes(report):
    notes = []
    dq, mq, reliability = report.get("data_quality", {}), report.get("movement_quality", {}), report.get("camera_reliability", {})
    notes.append(f"Data Quality: {dq.get('grade')} ({dq.get('score')}/100). Movement: {mq.get('grade')} ({mq.get('score')}/100).")
    notes.extend(mq.get("flags", []))
    notes.extend(dq.get("reasons", []))
    return notes

# ==================================================
# PDF GENERATOR
# ==================================================
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
            c.showPage()
            y = height - 0.75 * inch
        c.drawImage(chart_path, 0.75 * inch, y - 2.5*inch, width=6.8 * inch, height=2.5 * inch, preserveAspectRatio=True)

    c.save()

# ==================================================
# VIDEO ANALYSIS ENGINE
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

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    current_frame, processed_frames, low_confidence_frames, movement_faults, valgus_errors = 0, 0, 0, 0, 0
    pelvic_history, knee_flexion_history, trunk_lean_history, shoulder_tilt_history = [], [], [], []

    preview = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    frame_stride = 2

    try:
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                current_frame += 1
                if current_frame % frame_stride != 0: continue
                processed_frames += 1

                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                    progress_text.text(f"{label} | Processing frame {current_frame} of {total_frames}")

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                warning_text, warning_color = "NO POSE DETECTED", (255, 165, 0)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    try:
                        r_hip, r_hip_vis = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                        r_knee, r_knee_vis = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE)
                        r_ankle, r_ankle_vis = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                        l_hip, l_hip_vis = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                        l_knee, l_knee_vis = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_KNEE)
                        l_ankle, l_ankle_vis = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                        r_shoulder, r_shoulder_vis = get_lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
                        l_shoulder, l_shoulder_vis = get_lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)

                        if not visible_enough(r_hip_vis, r_knee_vis, r_ankle_vis, l_hip_vis, l_knee_vis, l_ankle_vis, r_shoulder_vis, l_shoulder_vis):
                            low_confidence_frames += 1
                            pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)
                        else:
                            r_knee_angle = calculate_angle_3d(r_hip, r_knee, r_ankle)
                            l_knee_angle = calculate_angle_3d(l_hip, l_knee, l_ankle)
                            knee_flexion = 180 - ((r_knee_angle + l_knee_angle) / 2)

                            pelvic_drop = calculate_pelvic_drop(l_hip, r_hip)
                            trunk_lean = calculate_trunk_lean(l_shoulder, r_shoulder, l_hip, r_hip)
                            shoulder_tilt = calculate_shoulder_tilt(l_shoulder, r_shoulder)
                            valgus_detected = detect_valgus(l_knee, r_knee, l_ankle, r_ankle)

                            if valgus_detected: valgus_errors += 1

                            pelvic_history.append(pelvic_drop)
                            knee_flexion_history.append(knee_flexion)
                            trunk_lean_history.append(trunk_lean)
                            shoulder_tilt_history.append(shoulder_tilt)

                            warning_text, warning_color, fault = evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected)
                            if fault: movement_faults += 1

                    except Exception:
                        low_confidence_frames += 1
                        pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    low_confidence_frames += 1
                    pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)

                preview.image(image, channels="RGB", use_container_width=True)
    finally:
        cap.release()
        try: os.remove(video_path)
        except Exception: pass

    preview.empty(); progress_text.empty(); progress_bar.empty()

    if processed_frames == 0: return None

    valid_data_frames = count_valid_values(knee_flexion_history)
    valid_frames_for_rates = max(valid_data_frames, 1)

    report = {
        "label": label, "movement_test": movement_test, "camera_view": camera_view, "client_profile": client_profile or {},
        "fps": fps, "total_frames": total_frames, "processed_frames": processed_frames, "valid_data_frames": valid_data_frames, "low_confidence_frames": low_confidence_frames,
        "max_pelvic_drop": safe_max(pelvic_history), "avg_pelvic_drop": safe_mean(pelvic_history),
        "max_knee_flexion": safe_max(knee_flexion_history), "avg_knee_flexion": safe_mean(knee_flexion_history),
        "max_trunk_lean": safe_max(trunk_lean_history), "avg_trunk_lean": safe_mean(trunk_lean_history),
        "max_shoulder_tilt": safe_max(shoulder_tilt_history), "avg_shoulder_tilt": safe_mean(shoulder_tilt_history),
        "valgus_rate": valgus_errors / valid_frames_for_rates * 100, "movement_fault_rate": movement_faults / valid_frames_for_rates * 100,
        "tracking_confidence_rate": valid_data_frames / processed_frames * 100,
        "pelvic_history": pelvic_history, "knee_flexion_history": knee_flexion_history, "trunk_lean_history": trunk_lean_history, "shoulder_tilt_history": shoulder_tilt_history,
    }

    report["metric_confidence"] = get_metric_confidence(movement_test, camera_view)
    report["camera_reliability"] = assess_camera_reliability(movement_test, camera_view, report["tracking_confidence_rate"])
    report["data_quality"] = assess_data_quality(report)
    report["movement_quality"] = assess_movement_quality(report)
    report["notes"] = generate_notes(report)

    return report


# ==================================================
# SQLite HISTORY
# ==================================================
def init_db():
    os.makedirs("reports", exist_ok=True)
    conn = sqlite3.connect("reports/report_history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, client_name TEXT, client_age TEXT,
            movement_test TEXT, camera_view TEXT, movement_score INTEGER
        )
    ''')
    conn.commit()
    return conn

def save_report_history(report):
    conn = init_db()
    cursor = conn.cursor()
    client = report.get("client_profile", {})
    mq = report.get("movement_quality", {})
    
    cursor.execute('''
        INSERT INTO history (date, client_name, client_age, movement_test, camera_view, movement_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        client.get("client_name", ""),
        client.get("client_age", ""),
        report.get("movement_test", ""),
        report.get("camera_view", ""),
        mq.get("score", 0)
    ))
    conn.commit()
    conn.close()

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
    else:
        st.info("No saved reports yet.")

# ==================================================
# DISPLAY
# ==================================================
def show_report(report):
    st.header(f"📊 Final Biomechanical Report: {report['label']}")
    st.info(f"Test: **{report['movement_test']}** | Camera: **{report.get('camera_view')}**")

    mq = report.get("movement_quality", {})
    col_mq1, col_mq2, col_mq3 = st.columns(3)
    col_mq1.metric("Movement Quality", f"{mq.get('grade')} ({mq.get('label')})")
    col_mq2.metric("Score", f"{mq.get('score')}/100")
    col_mq3.metric("Limitation", mq.get("primary_limitation"))

    st.markdown("---")
    st.subheader("📈 Kinematic Data")
    chart_df = pd.DataFrame({
        "Pelvic Drop": report["pelvic_history"],
        "Knee Flexion": report["knee_flexion_history"],
        "Trunk Lean": report["trunk_lean_history"]
    }).apply(pd.to_numeric, errors="coerce")
    st.line_chart(chart_df)

    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    save_chart_png(chart_df, f"{report['movement_test']} Kinematic Chart", chart_path)
    create_pdf_report(report, chart_path, pdf_path)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("Download PDF Report", pdf_file, "report.pdf", "application/pdf")
    with col_b:
        if st.button(f"Save {report['label']} to History", key=f"save_{report['label']}"):
            save_report_history(report)
            st.success("Saved!")

def compare_reports(before, after):
    st.header("🔁 Side-by-Side Comparison")
    st.info(f"Comparison Mode: {before['movement_test']}")

    st.subheader("Time-Normalized Comparison Charts")
    compare_df = pd.DataFrame({
        "Before Knee Flexion": normalize_time_series(before["knee_flexion_history"]),
        "After Knee Flexion": normalize_time_series(after["knee_flexion_history"]),
        "Before Trunk Lean": normalize_time_series(before["trunk_lean_history"]),
        "After Trunk Lean": normalize_time_series(after["trunk_lean_history"]),
    })
    st.line_chart(compare_df)

# ==================================================
# MAIN UI
# ==================================================
analysis_type = st.radio("Choose Analysis Type", ["Single Video Analysis", "Before / After Comparison"], horizontal=True)
movement_test = st.selectbox("Choose Movement Test", list(MOVEMENT_TESTS.keys()))
camera_view = st.selectbox("Choose Camera View", list(CAMERA_VIEWS.keys()))

if analysis_type == "Single Video Analysis":
    uploaded_video = st.file_uploader("Upload Movement Video", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        # Check Session State to prevent re-running
        if st.session_state.single_video_name != uploaded_video.name:
            st.session_state.single_report = analyze_video(
                uploaded_video, movement_test, camera_view, "Single Video Report", client_profile
            )
            st.session_state.single_video_name = uploaded_video.name

        if st.session_state.single_report:
            show_report(st.session_state.single_report)

else:
    col1, col2 = st.columns(2)
    with col1: before_video = st.file_uploader("Upload BEFORE Video", type=["mp4", "mov", "avi"], key="before")
    with col2: after_video = st.file_uploader("Upload AFTER Video", type=["mp4", "mov", "avi"], key="after")

    if before_video and after_video:
        if st.session_state.before_video_name != before_video.name:
            st.info("Processing BEFORE video...")
            st.session_state.before_report = analyze_video(before_video, movement_test, camera_view, "Before Report", client_profile)
            st.session_state.before_video_name = before_video.name
            
        if st.session_state.after_video_name != after_video.name:
            st.info("Processing AFTER video...")
            st.session_state.after_report = analyze_video(after_video, movement_test, camera_view, "After Report", client_profile)
            st.session_state.after_video_name = after_video.name

        if st.session_state.before_report and st.session_state.after_report:
            show_report(st.session_state.before_report)
            show_report(st.session_state.after_report)
            compare_reports(st.session_state.before_report, st.session_state.after_report)

show_history_dashboard()
