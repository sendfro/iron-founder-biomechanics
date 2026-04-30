import os
import json
import tempfile
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


# -------------------------------------------------
# APP SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Iron Founder Biomechanics v3.0",
    layout="wide"
)

st.title("Iron Founder AI: Motion Capture Engine v3.0")
st.markdown("Upload movement videos for AI-assisted biomechanical screening and coaching cues.")

st.warning(
    "This tool provides AI-assisted movement screening, not medical diagnosis. "
    "For pain, injury, or clinical decisions, consult a qualified professional."
)


# -------------------------------------------------
# CLIENT PROFILE SIDEBAR
# -------------------------------------------------
st.sidebar.header("Client Profile")

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


# -------------------------------------------------
# MOVEMENT TEST SETTINGS
# -------------------------------------------------
MOVEMENT_TESTS = {
    "Squat Analysis": {
        "description": "Analyzes knee flexion depth, knee valgus risk, trunk lean, and pelvic control during squat movement.",
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


# -------------------------------------------------
# CAMERA VIEW SETTINGS
# -------------------------------------------------
CAMERA_VIEWS = {
    "Front View": {
        "description": "Best for knee valgus, shoulder tilt, pelvic tilt, and left/right symmetry.",
        "best_for": ["Knee valgus", "Shoulder tilt", "Pelvic tilt", "Left/right asymmetry"],
        "weak_for": ["True squat depth", "Forward trunk lean", "Precise knee flexion"],
    },
    "Side View": {
        "description": "Best for squat depth, knee flexion, hip hinge, trunk lean, and landing mechanics.",
        "best_for": ["Knee flexion", "Squat depth", "Trunk lean", "Jump landing absorption"],
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


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_lm(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return [lm.x, lm.y], lm.visibility


def visible_enough(*scores, threshold=0.55):
    return all(score >= threshold for score in scores)


def safe_max(values):
    clean_values = [v for v in values if v is not None and not pd.isna(v)]
    return max(clean_values) if clean_values else 0


def safe_mean(values):
    clean_values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean_values)) if clean_values else 0


def count_valid_values(values):
    return len([v for v in values if v is not None and not pd.isna(v)])


def pad_list(values, target_length):
    padded = list(values)
    while len(padded) < target_length:
        padded.append(None)
    return padded


def calculate_line_tilt(point_a, point_b):
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

    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return angle


def detect_valgus(left_knee, right_knee, left_ankle, right_ankle):
    knee_dist = np.linalg.norm(np.array(right_knee) - np.array(left_knee))
    ankle_dist = np.linalg.norm(np.array(right_ankle) - np.array(left_ankle))

    if ankle_dist == 0:
        return False
    return knee_dist < ankle_dist * 0.8


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


def draw_wrapped_text(c, text, x, y, max_chars=90, line_height=0.18 * inch):
    if not text:
        return y

    words = str(text).split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line + " " + word) <= max_chars:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    for line in lines:
        c.drawString(x, y, line)
        y -= line_height

    return y


# -------------------------------------------------
# DATA QUALITY (TRUST GRADE) ENGINE
# -------------------------------------------------
def assess_camera_reliability(movement_test, camera_view, tracking_confidence_rate):
    score = 100
    warnings = []
    strengths = []

    if tracking_confidence_rate < 70:
        score -= 35
        warnings.append("Tracking confidence low. Ensure full body is visible with good lighting.")
    elif tracking_confidence_rate < 85:
        score -= 15
        warnings.append("Tracking confidence moderate. Interpret carefully.")
    else:
        strengths.append("Tracking confidence strong.")

    if movement_test == "Squat Analysis":
        if camera_view == "Side View":
            strengths.append("Side view is strong for squat depth and trunk lean.")
            score -= 5
        elif camera_view == "Front View":
            strengths.append("Front view is strong for detecting knee valgus.")
            score -= 10
        elif camera_view == "Rear View":
            warnings.append("Rear view is not ideal for squat analysis.")
            score -= 25
        else:
            score -= 35

    elif movement_test == "Running / Gait Analysis":
        if camera_view == "Rear View": strengths.append("Rear view is useful for pelvic drop.")
        elif camera_view == "Front View": score -= 10
        elif camera_view == "Side View": score -= 15
        else: score -= 35

    elif movement_test == "Jump Landing":
        if camera_view == "Front View": score -= 5
        elif camera_view == "Side View": score -= 5
        elif camera_view == "Rear View": score -= 25
        else: score -= 35

    elif movement_test == "Posture Screen":
        if camera_view == "Front View": strengths.append("Front view is strong for posture screening.")
        elif camera_view == "Side View": score -= 15
        elif camera_view == "Rear View": score -= 5
        else: score -= 35

    score = max(0, min(100, score))
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
    elif tracking_score >= 80: score -= 8; reasons.append("Pose tracking was good.")
    elif tracking_score >= 70: score -= 16; reasons.append("Pose tracking was moderate.")
    else: score -= 32; reasons.append("Pose tracking was low.")

    if processed_frames < 20: score -= 25; reasons.append("Very few frames processed.")
    elif processed_frames < 50: score -= 12; reasons.append("Limited frame sample.")

    valid_ratio = valid_data_frames / max(processed_frames, 1)
    if valid_ratio < 0.9 and valid_ratio >= 0.75: score -= 8
    elif valid_ratio < 0.75 and valid_ratio >= 0.6: score -= 16
    elif valid_ratio < 0.6: score -= 30

    score = max(0, min(100, score))
    grade = "A+" if score >= 95 else "A" if score >= 90 else "B+" if score >= 85 else "B" if score >= 80 else "C+" if score >= 75 else "C" if score >= 70 else "D" if score >= 60 else "Low Trust"

    return {"score": score, "grade": grade, "reasons": reasons}


# -------------------------------------------------
# V3.0 MOVEMENT QUALITY ENGINE
# -------------------------------------------------
def generate_movement_quality_v3(report):
    test = report["movement_test"]
    valgus = report["valgus_rate"]
    flexion = report["max_knee_flexion"]
    trunk = report["max_trunk_lean"]
    pelvic = report["max_pelvic_drop"]
    shoulder = report["max_shoulder_tilt"]
    settings = MOVEMENT_TESTS[test]

    score = 100
    limitations = {}
    flags = []
    cues = []

    if test == "Squat Analysis":
        if flexion < settings["knee_flexion_target"]:
            diff = settings["knee_flexion_target"] - flexion
            score -= min(diff * 1.5, 30)
            limitations["Limited Squat Depth"] = diff
            flags.append("Range of Motion Restriction (Depth)")
            cues.append("Focus on reaching parallel; elevate heels if ankle mobility is restricted.")

        if valgus > 5:
            score -= min(valgus, 35)
            limitations["Knee Valgus (Inward Cave)"] = valgus
            flags.append(f"Joint Risk: {valgus:.1f}% Valgus Detected")
            cues.append("Root your feet firmly and actively press knees outward against an imaginary band.")

        if trunk > settings["trunk_lean_limit"]:
            diff = trunk - settings["trunk_lean_limit"]
            score -= min(diff * 1.5, 20)
            limitations["Excessive Trunk Lean"] = diff
            flags.append("Lower Back Shear Risk (Forward Lean)")
            cues.append("Keep your chest proud and core fully braced throughout the descent.")

    elif test == "Running / Gait Analysis":
        if pelvic > settings["pelvic_drop_limit"]:
            score -= min((pelvic - settings["pelvic_drop_limit"]) * 3, 30)
            limitations["Pelvic Instability (Drop)"] = pelvic
            flags.append("Trendelenburg Sign / Hip Abductor Weakness")
            cues.append("Engage glutes to keep hips level on impact; consider lateral band walks.")

        if trunk > settings["trunk_lean_limit"]:
            score -= min((trunk - settings["trunk_lean_limit"]) * 2, 20)
            limitations["Excessive Trunk Lean"] = trunk
            flags.append("Inefficient Running Posture")
            cues.append("Run tall with a slight forward lean from the ankles, not the waist.")

    elif test == "Jump Landing":
        if flexion < settings["landing_knee_flexion_min"]:
            score -= min((settings["landing_knee_flexion_min"] - flexion) * 2, 30)
            limitations["Stiff Landing Mechanics"] = flexion
            flags.append("High Impact Force Absorption Risk")
            cues.append("Land softly like a ninja; sink into the hips and knees to absorb force.")

        if valgus > 5:
            score -= min(valgus, 40)
            limitations["Dynamic Knee Valgus"] = valgus
            flags.append(f"High ACL Risk: {valgus:.1f}% Valgus on Landing")
            cues.append("Stick the landing with knees tracking directly over the second toe.")

    elif test == "Posture Screen":
        if shoulder > settings["shoulder_tilt_limit"]:
            score -= min((shoulder - settings["shoulder_tilt_limit"]) * 3, 25)
            limitations["Shoulder Asymmetry"] = shoulder
            flags.append("Upper Body Imbalance")
            cues.append("Check for carrying heavy loads on one side; stretch upper traps.")

        if pelvic > settings["pelvic_drop_limit"]:
            score -= min((pelvic - settings["pelvic_drop_limit"]) * 3, 25)
            limitations["Pelvic Tilt / Asymmetry"] = pelvic
            flags.append("Lower Body Imbalance")
            cues.append("Check for leg length discrepancy or isolated glute medius weakness.")

    # Determine Primary Limitation
    primary_limitation = "None Detected"
    if limitations:
        primary_limitation = max(limitations, key=limitations.get)

    # Defaults & formatting
    if not cues:
        cues.append("Maintain current form and utilize progressive overload.")
    cues = cues[:3]

    if not flags:
        flags.append("No major movement risks detected.")

    score = max(0, min(100, score))
    if score >= 90: grade = "A (Excellent)"
    elif score >= 80: grade = "B (Good)"
    elif score >= 70: grade = "C (Fair)"
    else: grade = "D (Needs Improvement)"

    # Recommended Retest View based on Camera Reliability
    retest_view = "Current camera view is acceptable."
    if report.get("camera_reliability", {}).get("score", 100) < 85:
        if test == "Squat Analysis": retest_view = "Use Side View for better depth/trunk tracking, or Front View for pure valgus tracking."
        elif test == "Running / Gait Analysis": retest_view = "Use Rear View (waist height) or strict Side View."
        elif test == "Jump Landing": retest_view = "Use Front View for valgus tracking, Side View for landing depth."
        elif test == "Posture Screen": retest_view = "Use strict Front View, camera at waist/chest height."

    return {
        "score": score,
        "grade": grade,
        "primary_limitation": primary_limitation,
        "risk_flags": flags,
        "coaching_cues": cues,
        "retest_view": retest_view
    }


# -------------------------------------------------
# PDF REPORT GENERATOR
# -------------------------------------------------
def create_pdf_report(report, chart_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    y = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.75 * inch, y, "Iron Founder AI: Movement Quality Report v3.0")
    y -= 0.35 * inch

    c.setFont("Helvetica", 12)
    c.drawString(0.75 * inch, y, f"Movement Test: {report['movement_test']} | View: {report.get('camera_view', 'N/A')}")
    y -= 0.25 * inch

    trust = report.get("data_quality", {})
    c.drawString(0.75 * inch, y, f"Data Quality Grade: {trust.get('grade', 'N/A')} ({trust.get('score', 0)}/100)")
    y -= 0.25 * inch
    c.drawString(0.75 * inch, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 0.45 * inch

    # MOVEMENT QUALITY V3 SECTION
    mq = report.get("movement_quality_v3", {})
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "Movement Quality Summary")
    y -= 0.25 * inch
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.9 * inch, y, f"Quality Grade: {mq.get('grade', 'N/A')} ({mq.get('score', 0)}/100)")
    y -= 0.2 * inch
    c.drawString(0.9 * inch, y, f"Primary Limitation: {mq.get('primary_limitation', 'N/A')}")
    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.9 * inch, y, "Risk Flags:")
    c.setFont("Helvetica", 10)
    y -= 0.2 * inch
    for flag in mq.get("risk_flags", []):
        y = draw_wrapped_text(c, f"• {flag}", 1.1 * inch, y, max_chars=85)
        y -= 0.05 * inch
    y -= 0.15 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(0.9 * inch, y, "Top Coaching Cues:")
    c.setFont("Helvetica", 10)
    y -= 0.2 * inch
    for cue in mq.get("coaching_cues", []):
        y = draw_wrapped_text(c, f"• {cue}", 1.1 * inch, y, max_chars=85)
        y -= 0.05 * inch
    y -= 0.25 * inch

    # KINEMATIC METRICS
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "Kinematic Metrics")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    rows = [
        ("Max Knee Flexion", f"{report['max_knee_flexion']:.1f}°"),
        ("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°"),
        ("Max Pelvic Drop", f"{report['max_pelvic_drop']:.1f}°"),
        ("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%"),
        ("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%"),
    ]

    for name, value in rows:
        c.drawString(0.9 * inch, y, f"{name}: {value}")
        y -= 0.2 * inch

    # CHART
    if os.path.exists(chart_path):
        if y < 3.5 * inch:
            c.showPage()
            y = height - 0.75 * inch
        y -= 2.9 * inch
        c.drawImage(chart_path, 0.75 * inch, y, width=6.8 * inch, height=2.5 * inch, preserveAspectRatio=True)

    c.setFont("Helvetica", 8)
    c.drawString(0.75 * inch, 0.5 * inch, "Disclaimer: AI-assisted movement screening only. Not medical advice.")
    c.save()


# -------------------------------------------------
# MODE-SPECIFIC EVALUATION (FOR CHARTING/VIDEO OVERLAY)
# -------------------------------------------------
def evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected):
    settings = MOVEMENT_TESTS[movement_test]
    warning_text = f"{movement_test.upper()}: FORM SOLID"
    warning_color = (0, 255, 0)
    fault_detected = False

    if movement_test == "Squat Analysis":
        warning_text = f"SQUAT | Knee Flexion: {knee_flexion:.1f}°"
        if knee_flexion < settings["knee_flexion_target"]:
            warning_text = "SQUAT WARNING: LIMITED DEPTH"; warning_color = (255, 165, 0); fault_detected = True
        if valgus_detected:
            warning_text = "SQUAT WARNING: KNEE VALGUS"; warning_color = (255, 0, 0); fault_detected = True

    elif movement_test == "Running / Gait Analysis":
        warning_text = f"GAIT | Pelvic Drop: {pelvic_drop:.1f}°"
        if pelvic_drop > settings["pelvic_drop_limit"]:
            warning_text = "GAIT WARNING: PELVIC DROP"; warning_color = (255, 165, 0); fault_detected = True

    elif movement_test == "Jump Landing":
        warning_text = f"LANDING | Knee Flexion: {knee_flexion:.1f}°"
        if valgus_detected:
            warning_text = "LANDING WARNING: KNEE VALGUS"; warning_color = (255, 0, 0); fault_detected = True

    elif movement_test == "Posture Screen":
        warning_text = f"POSTURE | Shoulder Tilt: {shoulder_tilt:.1f}°"
        if shoulder_tilt > settings["shoulder_tilt_limit"]:
            warning_text = "POSTURE WARNING: SHOULDER TILT"; warning_color = (255, 165, 0); fault_detected = True

    return warning_text, warning_color, fault_detected


# -------------------------------------------------
# CORE VIDEO ANALYSIS
# -------------------------------------------------
def analyze_video(uploaded_file, movement_test, camera_view, label="Video", client_profile=None):
    suffix = os.path.splitext(uploaded_file.name)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
        tfile.write(uploaded_file.read())
        video_path = tfile.name

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Could not open {label}. Try another MP4, MOV, or AVI file.")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        current_frame = 0; processed_frames = 0; low_confidence_frames = 0; movement_faults = 0; valgus_errors = 0
        pelvic_history = []; knee_flexion_history = []; trunk_lean_history = []; shoulder_tilt_history = []

        preview = st.empty(); progress_text = st.empty(); progress_bar = st.progress(0)
        frame_stride = 2

        with mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                current_frame += 1
                if current_frame % frame_stride != 0: continue
                processed_frames += 1

                if total_frames > 0:
                    progress_bar.progress(min(current_frame / total_frames, 1.0))
                    progress_text.text(f"{label} | {movement_test}: Processing frame {current_frame} of {total_frames}")

                # CLAHE LIGHTING ENHANCEMENT
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                warning_text = "NO POSE DETECTED"; warning_color = (255, 165, 0)

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
                            warning_text = "LOW CONFIDENCE LANDMARKS"; warning_color = (255, 165, 0)
                        else:
                            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                            knee_flexion = 180 - ((r_knee_angle + l_knee_angle) / 2)
                            pelvic_drop = calculate_pelvic_drop(l_hip, r_hip)
                            trunk_lean = calculate_trunk_lean(l_shoulder, r_shoulder, l_hip, r_hip)
                            shoulder_tilt = calculate_shoulder_tilt(l_shoulder, r_shoulder)
                            valgus_detected = detect_valgus(l_knee, r_knee, l_ankle, r_ankle)

                            if valgus_detected: valgus_errors += 1
                            pelvic_history.append(pelvic_drop); knee_flexion_history.append(knee_flexion); trunk_lean_history.append(trunk_lean); shoulder_tilt_history.append(shoulder_tilt)

                            warning_text, warning_color, fault_detected = evaluate_frame_by_test(movement_test, knee_flexion, pelvic_drop, trunk_lean, shoulder_tilt, valgus_detected)
                            if fault_detected: movement_faults += 1
                    except Exception:
                        low_confidence_frames += 1
                        pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)
                        warning_text = "FRAME SKIPPED"; warning_color = (255, 165, 0)

                    cv2.putText(image, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=warning_color, thickness=2, circle_radius=2))
                else:
                    low_confidence_frames += 1
                    pelvic_history.append(None); knee_flexion_history.append(None); trunk_lean_history.append(None); shoulder_tilt_history.append(None)
                    cv2.putText(image, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2, cv2.LINE_AA)

                preview.image(image, channels="RGB", use_container_width=True)

        cap.release(); preview.empty(); progress_text.empty(); progress_bar.empty()

        if processed_frames == 0: return None

        valid_data_frames = count_valid_values(knee_flexion_history)
        valid_frames_for_rates = max(valid_data_frames, 1)

        report = {
            "label": label, "movement_test": movement_test, "camera_view": camera_view, "client_profile": client_profile or {},
            "fps": fps, "total_frames": total_frames, "processed_frames": processed_frames,
            "valid_data_frames": valid_data_frames, "low_confidence_frames": low_confidence_frames,
            "max_pelvic_drop": safe_max(pelvic_history), "avg_pelvic_drop": safe_mean(pelvic_history),
            "max_knee_flexion": safe_max(knee_flexion_history), "avg_knee_flexion": safe_mean(knee_flexion_history),
            "max_trunk_lean": safe_max(trunk_lean_history), "avg_trunk_lean": safe_mean(trunk_lean_history),
            "max_shoulder_tilt": safe_max(shoulder_tilt_history), "avg_shoulder_tilt": safe_mean(shoulder_tilt_history),
            "valgus_rate": valgus_errors / valid_frames_for_rates * 100,
            "movement_fault_rate": movement_faults / valid_frames_for_rates * 100,
            "tracking_confidence_rate": valid_data_frames / processed_frames * 100,
            "pelvic_history": pelvic_history, "knee_flexion_history": knee_flexion_history,
            "trunk_lean_history": trunk_lean_history, "shoulder_tilt_history": shoulder_tilt_history,
        }

        report["camera_reliability"] = assess_camera_reliability(movement_test, camera_view, report["tracking_confidence_rate"])
        report["data_quality"] = assess_data_quality(report)
        report["movement_quality_v3"] = generate_movement_quality_v3(report)

        return report

    finally:
        try:
            if os.path.exists(video_path): os.remove(video_path)
        except Exception: pass


# -------------------------------------------------
# REPORT HISTORY
# -------------------------------------------------
def save_report_history(report):
    os.makedirs("reports", exist_ok=True)
    history_path = "reports/report_history.csv"

    client = report.get("client_profile", {})
    dq = report.get("data_quality", {})
    mq = report.get("movement_quality_v3", {})

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "client_name": client.get("client_name", ""),
        "client_activity": client.get("client_activity", ""),
        "movement_test": report.get("movement_test", ""),
        
        # V3 Data Additions
        "data_quality_grade": dq.get("grade", ""),
        "movement_quality_grade": mq.get("grade", ""),
        "movement_quality_score": mq.get("score", 0),
        "primary_limitation": mq.get("primary_limitation", ""),
        "coaching_cues": " | ".join(mq.get("coaching_cues", [])),
        
        "max_pelvic_drop": report.get("max_pelvic_drop", 0),
        "max_knee_flexion": report.get("max_knee_flexion", 0),
        "max_trunk_lean": report.get("max_trunk_lean", 0),
        "valgus_rate": report.get("valgus_rate", 0),
        "tracking_confidence": report.get("tracking_confidence_rate", 0),
    }

    new_row_df = pd.DataFrame([row])
    final_df = pd.concat([pd.read_csv(history_path), new_row_df], ignore_index=True) if os.path.exists(history_path) else new_row_df
    final_df.to_csv(history_path, index=False)


def show_history_dashboard():
    st.markdown("---")
    st.header("Client Report History")
    history_path = "reports/report_history.csv"

    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        search_name = st.text_input("Search Client Name")
        if search_name: history_df = history_df[history_df["client_name"].astype(str).str.contains(search_name, case=False, na=False)]
        
        st.dataframe(history_df, use_container_width=True)
        st.download_button("Download Full History CSV", data=history_df.to_csv(index=False).encode("utf-8"), file_name="iron_founder_history.csv", mime="text/csv")
    else:
        st.info("No saved reports yet.")


# -------------------------------------------------
# DISPLAY REPORT
# -------------------------------------------------
def show_report(report):
    st.header(f"📊 Final Biomechanical Report: {report['label']}")
    st.caption(f"Movement Test: **{report['movement_test']}** | Camera View: **{report.get('camera_view', 'Not selected')}**")

    # V3.0 MOVEMENT QUALITY UI
    mq = report["movement_quality_v3"]
    
    st.markdown("### 🏆 Movement Quality v3.0")
    colA, colB, colC = st.columns(3)
    colA.metric("Movement Quality Grade", f"{mq['grade']}")
    colB.metric("Quality Score", f"{mq['score']}/100")
    colC.metric("Primary Limitation", mq['primary_limitation'])

    with st.container():
        st.markdown("#### 🚨 Risk Flags")
        for flag in mq["risk_flags"]:
            st.error(f"**{flag}**")
            
        st.markdown("#### 🧠 Top Coaching Cues")
        for cue in mq["coaching_cues"]:
            st.success(f"💡 {cue}")
            
        if "acceptable" not in mq["retest_view"]:
            st.warning(f"🎥 **Recommended Retest View:** {mq['retest_view']}")

    st.markdown("---")

    # DATA QUALITY
    dq = report.get("data_quality", {})
    st.markdown("### 📡 Data & Tracking Quality")
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Quality Grade", dq.get("grade", "N/A"))
    col2.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")
    col3.metric("Valid Frames Processed", f"{report['valid_data_frames']}")
    
    with st.expander("View Data Quality Details", expanded=False):
        for reason in dq.get("reasons", []): st.write(f"• {reason}")

    st.markdown("---")

    # KINEMATICS
    st.markdown("### 📐 Kinematic Data")
    if report["movement_test"] == "Squat Analysis":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Knee Flexion", f"{report['max_knee_flexion']:.1f}°")
        c2.metric("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%")
        c3.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°")
        c4.metric("Max Pelvic Drop", f"{report['max_pelvic_drop']:.1f}°")

    chart_df = pd.DataFrame({"Pelvic Drop": report["pelvic_history"], "Knee Flexion": report["knee_flexion_history"], "Trunk Lean": report["trunk_lean_history"]}).apply(pd.to_numeric, errors="coerce")
    st.line_chart(chart_df)

    st.markdown("---")

    # EXPORT
    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    save_chart_png(chart_df, f"{report['movement_test']} Kinematics", chart_path)
    create_pdf_report(report, chart_path, pdf_path)

    col_a, col_b, col_c = st.columns(3)
    safe_name = report["movement_test"].replace(" ", "_").lower()

    with col_a:
        with open(chart_path, "rb") as img_file: st.download_button("Download Chart PNG", data=img_file, file_name=f"{safe_name}_chart.png", mime="image/png")
    with col_b:
        with open(pdf_path, "rb") as pdf_file: st.download_button("Download PDF Report", data=pdf_file, file_name=f"{safe_name}_report.pdf", mime="application/pdf")
    with col_c:
        if st.button(f"Save {report['label']} to History"):
            save_report_history(report)
            st.success("Report saved.")


# -------------------------------------------------
# COMPARISON DISPLAY
# -------------------------------------------------
def compare_reports(before, after):
    st.header("🔁 V3.0 Side-by-Side Comparison")
    
    st.markdown("### 🏆 Movement Quality Change")
    col1, col2, col3 = st.columns(3)
    col1.metric("Before Grade", before["movement_quality_v3"]["grade"])
    col2.metric("After Grade", after["movement_quality_v3"]["grade"])
    
    score_change = after["movement_quality_v3"]["score"] - before["movement_quality_v3"]["score"]
    col3.metric("Score Change", f"{score_change:+.1f} Points", delta=score_change)

    comparison = pd.DataFrame({
        "Metric": ["Primary Limitation", "Max Knee Flexion", "Max Trunk Lean", "Knee Valgus Risk", "Data Quality Grade"],
        "Before": [before["movement_quality_v3"]["primary_limitation"], f"{before['max_knee_flexion']:.1f}°", f"{before['max_trunk_lean']:.1f}°", f"{before['valgus_rate']:.1f}%", before["data_quality"]["grade"]],
        "After": [after["movement_quality_v3"]["primary_limitation"], f"{after['max_knee_flexion']:.1f}°", f"{after['max_trunk_lean']:.1f}°", f"{after['valgus_rate']:.1f}%", after["data_quality"]["grade"]],
    })
    st.dataframe(comparison, use_container_width=True)

    max_len = max(len(before["knee_flexion_history"]), len(after["knee_flexion_history"]))
    compare_df = pd.DataFrame({
        "Before Knee Flexion": pad_list(before["knee_flexion_history"], max_len),
        "After Knee Flexion": pad_list(after["knee_flexion_history"], max_len),
    }).apply(pd.to_numeric, errors="coerce")
    st.line_chart(compare_df)


# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
analysis_type = st.radio("Choose Analysis Type", ["Single Video Analysis", "Before / After Comparison"], horizontal=True)
movement_test = st.selectbox("Choose Movement Test", ["Squat Analysis", "Running / Gait Analysis", "Jump Landing", "Posture Screen"])
camera_view = st.selectbox("Choose Camera View", ["Front View", "Side View", "Rear View", "Diagonal / Unknown"])

if analysis_type == "Single Video Analysis":
    uploaded_video = st.file_uploader("Upload Movement Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        report = analyze_video(uploaded_video, movement_test, camera_view, "Single Video Report", client_profile)
        if report: show_report(report)

else:
    col1, col2 = st.columns(2)
    with col1: before_video = st.file_uploader("Upload BEFORE Video", type=["mp4", "mov", "avi"], key="before")
    with col2: after_video = st.file_uploader("Upload AFTER Video", type=["mp4", "mov", "avi"], key="after")

    if before_video and after_video:
        st.info("Processing BEFORE video...")
        before_report = analyze_video(before_video, movement_test, camera_view, "Before Report", client_profile)
        st.info("Processing AFTER video...")
        after_report = analyze_video(after_video, movement_test, camera_view, "After Report", client_profile)
        if before_report and after_report:
            show_report(before_report)
            show_report(after_report)
            compare_reports(before_report, after_report)

show_history_dashboard()
