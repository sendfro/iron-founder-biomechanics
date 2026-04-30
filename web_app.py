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
    page_title="Iron Founder Biomechanics",
    layout="wide"
)

st.title("Iron Founder AI: Motion Capture Engine")
st.markdown("Upload movement videos for AI-assisted biomechanical screening.")

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
        "best_for": [
            "Knee valgus",
            "Shoulder tilt",
            "Pelvic tilt",
            "Left/right asymmetry",
        ],
        "weak_for": [
            "True squat depth",
            "Forward trunk lean",
            "Precise knee flexion",
        ],
    },
    "Side View": {
        "description": "Best for squat depth, knee flexion, hip hinge, trunk lean, and landing mechanics.",
        "best_for": [
            "Knee flexion",
            "Squat depth",
            "Trunk lean",
            "Jump landing absorption",
        ],
        "weak_for": [
            "Knee valgus",
            "Left/right asymmetry",
            "Pelvic drop",
        ],
    },
    "Rear View": {
        "description": "Useful for gait, pelvic drop, heel path, and left/right control from behind.",
        "best_for": [
            "Pelvic drop",
            "Gait symmetry",
            "Rear-chain control",
            "Foot path observation",
        ],
        "weak_for": [
            "Precise knee flexion",
            "Squat depth",
            "Forward trunk lean",
        ],
    },
    "Diagonal / Unknown": {
        "description": "Least reliable. Diagonal angles distort joint measurements and symmetry readings.",
        "best_for": [
            "General visual screening only",
        ],
        "weak_for": [
            "Knee flexion",
            "Knee valgus",
            "Pelvic drop",
            "Trunk lean",
            "Shoulder tilt",
        ],
    },
}


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(
        c[1] - b[1],
        c[0] - b[0]
    ) - np.arctan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle = abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_lm(landmarks, landmark_enum):
    lm = landmarks[landmark_enum.value]
    return [lm.x, lm.y], lm.visibility


def visible_enough(*scores, threshold=0.55):
    return all(score >= threshold for score in scores)


def calculate_line_tilt(point_a, point_b):
    radians = np.arctan2(
        point_b[1] - point_a[1],
        point_b[0] - point_a[0]
    )

    angle = abs(radians * 180.0 / np.pi)

    return min(angle, abs(180 - angle))


def calculate_pelvic_drop(left_hip, right_hip):
    return calculate_line_tilt(left_hip, right_hip)


def calculate_shoulder_tilt(left_shoulder, right_shoulder):
    return calculate_line_tilt(left_shoulder, right_shoulder)


def calculate_trunk_lean(left_shoulder, right_shoulder, left_hip, right_hip):
    mid_shoulder = [
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    ]

    mid_hip = [
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    ]

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
        plt.plot(dataframe[column], label=column)

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
# CAMERA RELIABILITY ENGINE
# -------------------------------------------------
def assess_camera_reliability(movement_test, camera_view, tracking_confidence_rate):
    score = 100
    warnings = []
    strengths = []

    if tracking_confidence_rate < 70:
        score -= 35
        warnings.append(
            "Tracking confidence was low. The body may not have been fully visible, lighting may have been poor, or landmarks were unstable."
        )
    elif tracking_confidence_rate < 85:
        score -= 15
        warnings.append(
            "Tracking confidence was moderate. Results are usable but should be interpreted carefully."
        )
    else:
        strengths.append(
            "Tracking confidence was strong, which improves report reliability."
        )

    if movement_test == "Squat Analysis":
        if camera_view == "Side View":
            strengths.append(
                "Side view is strong for squat depth, knee flexion, and trunk lean."
            )
            warnings.append(
                "Side view is weak for knee valgus. Use front view if valgus is the main concern."
            )
            score -= 5

        elif camera_view == "Front View":
            strengths.append(
                "Front view is strong for detecting knee valgus and left/right symmetry."
            )
            warnings.append(
                "Front view is weak for true squat depth and forward trunk lean. Use side view for better depth measurement."
            )
            score -= 10

        elif camera_view == "Rear View":
            warnings.append(
                "Rear view is not ideal for squat analysis. Use front view for valgus or side view for depth."
            )
            score -= 25

        else:
            warnings.append(
                "Diagonal or unknown camera angle makes squat metrics less reliable."
            )
            score -= 35

    elif movement_test == "Running / Gait Analysis":
        if camera_view == "Rear View":
            strengths.append(
                "Rear view is useful for pelvic drop, gait symmetry, and left/right control."
            )

        elif camera_view == "Front View":
            strengths.append(
                "Front view can help screen pelvic control and symmetry."
            )
            warnings.append(
                "Front view is less ideal than rear view for gait mechanics."
            )
            score -= 10

        elif camera_view == "Side View":
            strengths.append(
                "Side view is useful for stride posture and trunk lean."
            )
            warnings.append(
                "Side view is weaker for pelvic drop and left/right gait symmetry."
            )
            score -= 15

        else:
            warnings.append(
                "Diagonal or unknown camera angle makes gait metrics less reliable."
            )
            score -= 35

    elif movement_test == "Jump Landing":
        if camera_view == "Front View":
            strengths.append(
                "Front view is strong for landing knee valgus and left/right collapse."
            )
            warnings.append(
                "Front view is weaker for landing depth and force absorption. Use side view for knee-flexion absorption."
            )
            score -= 5

        elif camera_view == "Side View":
            strengths.append(
                "Side view is strong for landing knee flexion, trunk lean, and force absorption."
            )
            warnings.append(
                "Side view is weak for knee valgus. Use front view if valgus is the main concern."
            )
            score -= 5

        elif camera_view == "Rear View":
            warnings.append(
                "Rear view is less reliable for jump landing analysis. Use front or side view."
            )
            score -= 25

        else:
            warnings.append(
                "Diagonal or unknown camera angle makes landing metrics less reliable."
            )
            score -= 35

    elif movement_test == "Posture Screen":
        if camera_view == "Front View":
            strengths.append(
                "Front view is strong for posture screening, shoulder tilt, pelvic tilt, and left/right alignment."
            )

        elif camera_view == "Side View":
            strengths.append(
                "Side view is useful for forward head position and trunk lean."
            )
            warnings.append(
                "Side view is weaker for shoulder tilt and left/right pelvic alignment."
            )
            score -= 15

        elif camera_view == "Rear View":
            strengths.append(
                "Rear view can help assess shoulder and pelvic asymmetry."
            )
            score -= 5

        else:
            warnings.append(
                "Diagonal or unknown camera angle makes posture screening less reliable."
            )
            score -= 35

    score = max(0, min(100, score))

    if score >= 85:
        label = "High"
    elif score >= 65:
        label = "Medium"
    else:
        label = "Low"

    return {
        "score": score,
        "label": label,
        "warnings": warnings,
        "strengths": strengths,
    }


# -------------------------------------------------
# PDF REPORT
# -------------------------------------------------
def create_pdf_report(report, chart_path, pdf_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    y = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.75 * inch, y, "Iron Founder AI")
    y -= 0.35 * inch

    c.setFont("Helvetica", 12)
    c.drawString(0.75 * inch, y, report["label"])
    y -= 0.25 * inch

    c.drawString(0.75 * inch, y, f"Movement Test: {report['movement_test']}")
    y -= 0.25 * inch

    c.drawString(0.75 * inch, y, f"Camera View: {report.get('camera_view', 'Not selected')}")
    y -= 0.25 * inch

    c.drawString(0.75 * inch, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 0.45 * inch

    client = report.get("client_profile", {})

    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "Client Profile")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)

    client_rows = [
        ("Client Name", client.get("client_name", "")),
        ("Client Age", client.get("client_age", "")),
        ("Sport / Activity", client.get("client_activity", "")),
        ("Coach / Trainer", client.get("coach_name", "")),
    ]

    has_client_info = False

    for name, value in client_rows:
        if value:
            has_client_info = True
            c.drawString(0.9 * inch, y, f"{name}: {value}")
            y -= 0.23 * inch

    if client.get("client_notes"):
        has_client_info = True
        y -= 0.1 * inch
        c.drawString(0.9 * inch, y, "Session Notes:")
        y -= 0.2 * inch
        y = draw_wrapped_text(
            c,
            client.get("client_notes", ""),
            0.9 * inch,
            y,
            max_chars=85,
            line_height=0.18 * inch,
        )

    if not has_client_info:
        c.drawString(0.9 * inch, y, "No client profile entered.")
        y -= 0.25 * inch

    y -= 0.3 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "Summary Metrics")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)

    reliability = report.get("camera_reliability", {})

    rows = [
        ("Max Pelvic Drop", f"{report['max_pelvic_drop']:.1f} degrees"),
        ("Average Pelvic Drop", f"{report['avg_pelvic_drop']:.1f} degrees"),
        ("Max Knee Flexion", f"{report['max_knee_flexion']:.1f} degrees"),
        ("Average Knee Flexion", f"{report['avg_knee_flexion']:.1f} degrees"),
        ("Max Trunk Lean", f"{report['max_trunk_lean']:.1f} degrees"),
        ("Average Trunk Lean", f"{report['avg_trunk_lean']:.1f} degrees"),
        ("Max Shoulder Tilt", f"{report['max_shoulder_tilt']:.1f} degrees"),
        ("Average Shoulder Tilt", f"{report['avg_shoulder_tilt']:.1f} degrees"),
        ("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%"),
        ("Movement Fault Rate", f"{report['movement_fault_rate']:.1f}%"),
        ("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%"),
        ("Camera Reliability", f"{reliability.get('label', 'Unknown')} - {reliability.get('score', 0)}/100"),
        ("Processed Frames", str(report["processed_frames"])),
    ]

    for name, value in rows:
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 10)

        c.drawString(0.9 * inch, y, f"{name}: {value}")
        y -= 0.23 * inch

    y -= 0.25 * inch

    if y < 2.0 * inch:
        c.showPage()
        y = height - 0.75 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, y, "AI Notes")
    y -= 0.3 * inch

    c.setFont("Helvetica", 9)

    for note in report["notes"]:
        if y < 1.5 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 9)

        y = draw_wrapped_text(
            c,
            f"- {note}",
            0.9 * inch,
            y,
            max_chars=95,
            line_height=0.18 * inch,
        )
        y -= 0.15 * inch

    if os.path.exists(chart_path):
        if y < 3.5 * inch:
            c.showPage()
            y = height - 0.75 * inch

        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.75 * inch, y, "Kinematic Chart")
        y -= 2.9 * inch

        c.drawImage(
            chart_path,
            0.75 * inch,
            y,
            width=6.8 * inch,
            height=2.5 * inch,
            preserveAspectRatio=True,
        )

    c.setFont("Helvetica", 8)
    c.drawString(
        0.75 * inch,
        0.5 * inch,
        "Disclaimer: AI-assisted movement screening only. Not medical advice."
    )

    c.save()


# -------------------------------------------------
# MODE-SPECIFIC EVALUATION
# -------------------------------------------------
def evaluate_frame_by_test(
    movement_test,
    knee_flexion,
    pelvic_drop,
    trunk_lean,
    shoulder_tilt,
    valgus_detected,
):
    settings = MOVEMENT_TESTS[movement_test]

    warning_text = f"{movement_test.upper()}: FORM SOLID"
    warning_color = (0, 255, 0)
    fault_detected = False

    if movement_test == "Squat Analysis":
        warning_text = f"SQUAT | Knee Flexion: {knee_flexion:.1f}°"

        if knee_flexion < settings["knee_flexion_target"]:
            warning_text = "SQUAT WARNING: LIMITED DEPTH"
            warning_color = (255, 165, 0)
            fault_detected = True

        if valgus_detected:
            warning_text = "SQUAT WARNING: POSSIBLE KNEE VALGUS"
            warning_color = (255, 0, 0)
            fault_detected = True

        if trunk_lean > settings["trunk_lean_limit"]:
            warning_text += " | TRUNK LEAN"
            warning_color = (255, 165, 0)
            fault_detected = True

    elif movement_test == "Running / Gait Analysis":
        warning_text = f"GAIT | Pelvic Drop: {pelvic_drop:.1f}°"

        if pelvic_drop > settings["pelvic_drop_limit"]:
            warning_text = "GAIT WARNING: PELVIC DROP"
            warning_color = (255, 165, 0)
            fault_detected = True

        if trunk_lean > settings["trunk_lean_limit"]:
            warning_text += " | TRUNK LEAN"
            warning_color = (255, 165, 0)
            fault_detected = True

    elif movement_test == "Jump Landing":
        warning_text = f"LANDING | Knee Flexion: {knee_flexion:.1f}°"

        if knee_flexion < settings["landing_knee_flexion_min"]:
            warning_text = "LANDING WARNING: STIFF LANDING"
            warning_color = (255, 165, 0)
            fault_detected = True

        if valgus_detected:
            warning_text = "LANDING WARNING: KNEE VALGUS"
            warning_color = (255, 0, 0)
            fault_detected = True

        if trunk_lean > settings["trunk_lean_limit"]:
            warning_text += " | TRUNK LEAN"
            warning_color = (255, 165, 0)
            fault_detected = True

    elif movement_test == "Posture Screen":
        warning_text = f"POSTURE | Shoulder Tilt: {shoulder_tilt:.1f}°"

        if shoulder_tilt > settings["shoulder_tilt_limit"]:
            warning_text = "POSTURE WARNING: SHOULDER TILT"
            warning_color = (255, 165, 0)
            fault_detected = True

        if pelvic_drop > settings["pelvic_drop_limit"]:
            warning_text += " | PELVIC TILT"
            warning_color = (255, 165, 0)
            fault_detected = True

        if trunk_lean > settings["trunk_lean_limit"]:
            warning_text += " | TRUNK LEAN"
            warning_color = (255, 165, 0)
            fault_detected = True

    return warning_text, warning_color, fault_detected


def generate_notes(report):
    movement_test = report["movement_test"]
    settings = MOVEMENT_TESTS[movement_test]
    notes = []

    if movement_test == "Squat Analysis":
        if report["max_knee_flexion"] < settings["knee_flexion_target"]:
            notes.append(
                "Squat depth appears limited. The athlete did not reach the selected knee-flexion target."
            )
        else:
            notes.append(
                "Squat depth reached the selected knee-flexion target."
            )

        if report["valgus_rate"] > 0:
            notes.append(
                "Possible knee valgus was detected during squat frames. Cue knees to track over toes."
            )
        else:
            notes.append(
                "No major knee valgus pattern was detected during the squat test."
            )

        if report["max_trunk_lean"] > settings["trunk_lean_limit"]:
            notes.append(
                "Trunk lean exceeded the selected squat threshold. This may suggest compensation, ankle mobility limitation, or fatigue."
            )
        else:
            notes.append(
                "Trunk lean stayed within the selected squat threshold."
            )

    elif movement_test == "Running / Gait Analysis":
        if report["max_pelvic_drop"] > settings["pelvic_drop_limit"]:
            notes.append(
                "Pelvic drop exceeded the selected gait threshold. This may suggest hip stability issues, fatigue, or poor camera angle."
            )
        else:
            notes.append(
                "Pelvic control stayed within the selected gait threshold."
            )

        if report["max_trunk_lean"] > settings["trunk_lean_limit"]:
            notes.append(
                "Trunk lean exceeded the selected gait threshold. This may indicate compensation or inefficient running posture."
            )
        else:
            notes.append(
                "Trunk lean stayed within the selected gait threshold."
            )

    elif movement_test == "Jump Landing":
        if report["max_knee_flexion"] < settings["landing_knee_flexion_min"]:
            notes.append(
                "Landing appeared stiff. Low knee flexion during landing may suggest poor force absorption."
            )
        else:
            notes.append(
                "Landing knee flexion showed better force absorption."
            )

        if report["valgus_rate"] > 0:
            notes.append(
                "Possible knee valgus was detected during landing. This may increase stress on the knee during high-impact movement."
            )
        else:
            notes.append(
                "No major landing valgus pattern was detected."
            )

        if report["max_trunk_lean"] > settings["trunk_lean_limit"]:
            notes.append(
                "Trunk lean exceeded the selected landing threshold. This may suggest poor landing control or fatigue."
            )
        else:
            notes.append(
                "Trunk position stayed within the selected landing threshold."
            )

    elif movement_test == "Posture Screen":
        if report["max_shoulder_tilt"] > settings["shoulder_tilt_limit"]:
            notes.append(
                "Shoulder tilt exceeded the selected posture threshold. This may suggest asymmetry, camera angle error, or postural compensation."
            )
        else:
            notes.append(
                "Shoulder alignment stayed within the selected posture threshold."
            )

        if report["max_pelvic_drop"] > settings["pelvic_drop_limit"]:
            notes.append(
                "Pelvic tilt exceeded the selected posture threshold. This may suggest asymmetry, stance imbalance, or camera angle error."
            )
        else:
            notes.append(
                "Pelvic alignment stayed within the selected posture threshold."
            )

        if report["max_trunk_lean"] > settings["trunk_lean_limit"]:
            notes.append(
                "Trunk lean exceeded the selected posture threshold."
            )
        else:
            notes.append(
                "Trunk alignment stayed within the selected posture threshold."
            )

    reliability = report.get("camera_reliability", {})
    reliability_label = reliability.get("label", "Unknown")
    reliability_score = reliability.get("score", 0)

    notes.append(
        f"Camera reliability rating: {reliability_label} ({reliability_score}/100)."
    )

    for warning in reliability.get("warnings", []):
        notes.append(warning)

    for strength in reliability.get("strengths", []):
        notes.append(strength)

    if report["tracking_confidence_rate"] < 70:
        notes.append(
            "Tracking confidence was low. Use a clear full-body video with good lighting, stable camera position, and minimal obstruction."
        )

    return notes


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

        current_frame = 0
        processed_frames = 0
        low_confidence_frames = 0
        movement_faults = 0
        valgus_errors = 0

        pelvic_history = []
        knee_flexion_history = []
        trunk_lean_history = []
        shoulder_tilt_history = []

        preview = st.empty()
        progress_text = st.empty()
        progress_bar = st.progress(0)

        frame_stride = 2

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                current_frame += 1

                if current_frame % frame_stride != 0:
                    continue

                processed_frames += 1

                if total_frames > 0:
                    progress = min(current_frame / total_frames, 1.0)
                    progress_bar.progress(progress)
                    progress_text.text(
                        f"{label} | {movement_test}: Processing frame {current_frame} of {total_frames}"
                    )

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True

                warning_text = "NO POSE DETECTED"
                warning_color = (255, 165, 0)

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

                        enough_visibility = visible_enough(
                            r_hip_vis, r_knee_vis, r_ankle_vis,
                            l_hip_vis, l_knee_vis, l_ankle_vis,
                            r_shoulder_vis, l_shoulder_vis,
                        )

                        if not enough_visibility:
                            low_confidence_frames += 1

                            pelvic_history.append(0)
                            knee_flexion_history.append(0)
                            trunk_lean_history.append(0)
                            shoulder_tilt_history.append(0)

                            warning_text = "LOW CONFIDENCE LANDMARKS"
                            warning_color = (255, 165, 0)

                        else:
                            r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
                            l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)

                            average_knee_angle = (r_knee_angle + l_knee_angle) / 2
                            knee_flexion = 180 - average_knee_angle

                            pelvic_drop = calculate_pelvic_drop(l_hip, r_hip)

                            trunk_lean = calculate_trunk_lean(
                                l_shoulder,
                                r_shoulder,
                                l_hip,
                                r_hip,
                            )

                            shoulder_tilt = calculate_shoulder_tilt(
                                l_shoulder,
                                r_shoulder,
                            )

                            valgus_detected = detect_valgus(
                                l_knee,
                                r_knee,
                                l_ankle,
                                r_ankle,
                            )

                            if valgus_detected:
                                valgus_errors += 1

                            pelvic_history.append(pelvic_drop)
                            knee_flexion_history.append(knee_flexion)
                            trunk_lean_history.append(trunk_lean)
                            shoulder_tilt_history.append(shoulder_tilt)

                            warning_text, warning_color, fault_detected = evaluate_frame_by_test(
                                movement_test=movement_test,
                                knee_flexion=knee_flexion,
                                pelvic_drop=pelvic_drop,
                                trunk_lean=trunk_lean,
                                shoulder_tilt=shoulder_tilt,
                                valgus_detected=valgus_detected,
                            )

                            if fault_detected:
                                movement_faults += 1

                    except Exception:
                        low_confidence_frames += 1

                        pelvic_history.append(0)
                        knee_flexion_history.append(0)
                        trunk_lean_history.append(0)
                        shoulder_tilt_history.append(0)

                        warning_text = "FRAME SKIPPED"
                        warning_color = (255, 165, 0)

                    cv2.putText(
                        image,
                        warning_text,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        warning_color,
                        2,
                        cv2.LINE_AA,
                    )

                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=warning_color, thickness=2, circle_radius=2),
                    )

                else:
                    low_confidence_frames += 1
                    pelvic_history.append(0)
                    knee_flexion_history.append(0)
                    trunk_lean_history.append(0)
                    shoulder_tilt_history.append(0)

                preview.image(image, channels="RGB", use_container_width=True)

        cap.release()

    finally:
        # Robust temp file cleanup guaranteed to execute even if the CV2 loop crashes
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            st.warning(f"Could not remove temporary video file: {e}")

    preview.empty()
    progress_text.empty()
    progress_bar.empty()

    if processed_frames == 0:
        return None

    valid_frames = max(processed_frames - low_confidence_frames, 1)

    tracking_confidence_rate = (
        (processed_frames - low_confidence_frames) / processed_frames * 100
    )

    report = {
        "label": label,
        "movement_test": movement_test,
        "camera_view": camera_view,
        "client_profile": client_profile or {},

        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "low_confidence_frames": low_confidence_frames,

        "max_pelvic_drop": max(pelvic_history) if pelvic_history else 0,
        "avg_pelvic_drop": float(np.mean(pelvic_history)) if pelvic_history else 0,

        "max_knee_flexion": max(knee_flexion_history) if knee_flexion_history else 0,
        "avg_knee_flexion": float(np.mean(knee_flexion_history)) if knee_flexion_history else 0,

        "max_trunk_lean": max(trunk_lean_history) if trunk_lean_history else 0,
        "avg_trunk_lean": float(np.mean(trunk_lean_history)) if trunk_lean_history else 0,

        "max_shoulder_tilt": max(shoulder_tilt_history) if shoulder_tilt_history else 0,
        "avg_shoulder_tilt": float(np.mean(shoulder_tilt_history)) if shoulder_tilt_history else 0,

        "valgus_rate": valgus_errors / valid_frames * 100,
        "movement_fault_rate": movement_faults / valid_frames * 100,
        "tracking_confidence_rate": tracking_confidence_rate,

        "pelvic_history": pelvic_history,
        "knee_flexion_history": knee_flexion_history,
        "trunk_lean_history": trunk_lean_history,
        "shoulder_tilt_history": shoulder_tilt_history,
    }

    report["camera_reliability"] = assess_camera_reliability(
        movement_test=movement_test,
        camera_view=camera_view,
        tracking_confidence_rate=report["tracking_confidence_rate"],
    )

    report["notes"] = generate_notes(report)

    return report


# -------------------------------------------------
# REPORT HISTORY
# -------------------------------------------------
def save_report_history(report):
    # ------------------------------------------------------------------------
    # DEPLOYMENT NOTE:
    # When deploying to a stateless environment, swap this local CSV writing 
    # logic with a cloud database insert (e.g. Supabase) so your data 
    # persists across redeploys.
    # ------------------------------------------------------------------------
    
    os.makedirs("reports", exist_ok=True)

    history_path = "reports/report_history.csv"

    client = report.get("client_profile", {})

    row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "client_name": client.get("client_name", ""),
        "client_age": client.get("client_age", ""),
        "client_activity": client.get("client_activity", ""),
        "coach_name": client.get("coach_name", ""),
        "movement_test": report.get("movement_test", ""),
        "camera_view": report.get("camera_view", ""),
        "camera_reliability_label": report.get("camera_reliability", {}).get("label", ""),
        "camera_reliability_score": report.get("camera_reliability", {}).get("score", 0),
        "label": report.get("label", ""),

        "max_pelvic_drop": report.get("max_pelvic_drop", 0),
        "avg_pelvic_drop": report.get("avg_pelvic_drop", 0),
        "max_knee_flexion": report.get("max_knee_flexion", 0),
        "avg_knee_flexion": report.get("avg_knee_flexion", 0),
        "max_trunk_lean": report.get("max_trunk_lean", 0),
        "avg_trunk_lean": report.get("avg_trunk_lean", 0),
        "max_shoulder_tilt": report.get("max_shoulder_tilt", 0),
        "avg_shoulder_tilt": report.get("avg_shoulder_tilt", 0),
        "valgus_rate": report.get("valgus_rate", 0),
        "movement_fault_rate": report.get("movement_fault_rate", 0),
        "tracking_confidence_rate": report.get("tracking_confidence_rate", 0),

        "client_notes": client.get("client_notes", ""),
        "ai_notes": json.dumps(report.get("notes", [])),
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(history_path):
        old_df = pd.read_csv(history_path)
        final_df = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        final_df = new_row_df

    final_df.to_csv(history_path, index=False)


def show_history_dashboard():
    st.markdown("---")
    st.header("Client Report History")

    history_path = "reports/report_history.csv"

    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)

        search_name = st.text_input("Search Client Name")

        if search_name:
            history_df = history_df[
                history_df["client_name"].astype(str).str.contains(
                    search_name,
                    case=False,
                    na=False,
                )
            ]

        st.dataframe(history_df, use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Full History CSV",
            data=csv_data,
            file_name="iron_founder_report_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved reports yet.")


# -------------------------------------------------
# DISPLAY REPORT
# -------------------------------------------------
def show_report(report):
    st.header(f"📊 Final Biomechanical Report: {report['label']}")

    st.info(
        f"Movement Test: **{report['movement_test']}** | "
        f"Camera View: **{report.get('camera_view', 'Not selected')}**"
    )

    st.caption(MOVEMENT_TESTS[report["movement_test"]]["description"])

    reliability = report.get("camera_reliability", {})
    reliability_label = reliability.get("label", "Unknown")
    reliability_score = reliability.get("score", 0)

    if reliability_label == "High":
        st.success(f"Camera Reliability: {reliability_label} — {reliability_score}/100")
    elif reliability_label == "Medium":
        st.warning(f"Camera Reliability: {reliability_label} — {reliability_score}/100")
    else:
        st.error(f"Camera Reliability: {reliability_label} — {reliability_score}/100")

    with st.expander("Camera Reliability Details", expanded=False):
        st.write(f"**Selected View:** {report.get('camera_view', 'Not selected')}")
        st.write(f"**Reliability Score:** {reliability_score}/100")

        if reliability.get("strengths"):
            st.write("**Strengths:**")
            for item in reliability.get("strengths", []):
                st.success(item)

        if reliability.get("warnings"):
            st.write("**Warnings:**")
            for item in reliability.get("warnings", []):
                st.warning(item)

    client = report.get("client_profile", {})

    if any(client.values()):
        with st.expander("Client Profile", expanded=True):
            col_a, col_b, col_c, col_d = st.columns(4)

            col_a.write(f"**Name:** {client.get('client_name', '') or '—'}")
            col_b.write(f"**Age:** {client.get('client_age', '') or '—'}")
            col_c.write(f"**Activity:** {client.get('client_activity', '') or '—'}")
            col_d.write(f"**Coach:** {client.get('coach_name', '') or '—'}")

            if client.get("client_notes"):
                st.write("**Session Notes:**")
                st.write(client.get("client_notes"))

    movement_test = report["movement_test"]

    if movement_test == "Squat Analysis":
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Max Knee Flexion", f"{report['max_knee_flexion']:.1f}°", "Depth target: 90°+")
        col2.metric("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%")
        col3.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°", "Target < 20°")
        col4.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")

    elif movement_test == "Running / Gait Analysis":
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Max Pelvic Drop", f"{report['max_pelvic_drop']:.1f}°", "Target < 8°")
        col2.metric("Movement Fault Rate", f"{report['movement_fault_rate']:.1f}%")
        col3.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°", "Target < 15°")
        col4.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")

    elif movement_test == "Jump Landing":
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Max Landing Flexion", f"{report['max_knee_flexion']:.1f}°", "Target: 35°+")
        col2.metric("Knee Valgus Risk", f"{report['valgus_rate']:.1f}%")
        col3.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°", "Target < 20°")
        col4.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")

    elif movement_test == "Posture Screen":
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Max Shoulder Tilt", f"{report['max_shoulder_tilt']:.1f}°", "Target < 6°")
        col2.metric("Max Pelvic Tilt", f"{report['max_pelvic_drop']:.1f}°", "Target < 6°")
        col3.metric("Max Trunk Lean", f"{report['max_trunk_lean']:.1f}°", "Target < 10°")
        col4.metric("Tracking Confidence", f"{report['tracking_confidence_rate']:.1f}%")

    st.markdown("---")

    st.subheader("📈 Kinematic Data")

    chart_df = pd.DataFrame({
        "Pelvic Drop": report["pelvic_history"],
        "Knee Flexion": report["knee_flexion_history"],
        "Trunk Lean": report["trunk_lean_history"],
        "Shoulder Tilt": report["shoulder_tilt_history"],
    })

    if movement_test == "Squat Analysis":
        display_df = chart_df[["Knee Flexion", "Trunk Lean", "Pelvic Drop"]]

    elif movement_test == "Running / Gait Analysis":
        display_df = chart_df[["Pelvic Drop", "Trunk Lean", "Knee Flexion"]]

    elif movement_test == "Jump Landing":
        display_df = chart_df[["Knee Flexion", "Trunk Lean", "Pelvic Drop"]]

    else:
        display_df = chart_df[["Shoulder Tilt", "Pelvic Drop", "Trunk Lean"]]

    st.line_chart(display_df)

    st.markdown("---")

    st.subheader("Engine Summary")

    for note in report["notes"]:
        warning_keywords = [
            "exceeded",
            "valgus",
            "limited",
            "stiff",
            "low",
            "warning",
            "poor",
            "weak",
            "less reliable",
            "not ideal",
            "diagonal",
            "unknown",
            "moderate",
        ]

        if any(word in note.lower() for word in warning_keywords):
            st.warning(note)
        else:
            st.success(note)

    st.markdown("---")

    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    save_chart_png(
        display_df,
        f"{report['movement_test']} Kinematic Chart",
        chart_path,
    )

    create_pdf_report(report, chart_path, pdf_path)

    col_a, col_b, col_c = st.columns(3)

    safe_name = report["movement_test"].replace(" ", "_").replace("/", "").lower()

    with col_a:
        with open(chart_path, "rb") as img_file:
            st.download_button(
                label="Download Chart PNG",
                data=img_file,
                file_name=f"{safe_name}_chart.png",
                mime="image/png",
            )

    with col_b:
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=f"{safe_name}_report.pdf",
                mime="application/pdf",
            )

    with col_c:
        if st.button(f"Save {report['label']} to History"):
            save_report_history(report)
            st.success("Report saved to client history.")


# -------------------------------------------------
# COMPARISON DISPLAY
# -------------------------------------------------
def compare_reports(before, after):
    st.header("🔁 Side-by-Side Comparison")

    st.info(
        f"Comparison Mode: **{before['movement_test']}** | "
        f"Camera View: **{before.get('camera_view', 'Not selected')}**"
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Camera Reliability",
            "Max Pelvic Drop",
            "Average Pelvic Drop",
            "Max Knee Flexion",
            "Average Knee Flexion",
            "Max Trunk Lean",
            "Average Trunk Lean",
            "Max Shoulder Tilt",
            "Average Shoulder Tilt",
            "Knee Valgus Risk",
            "Movement Fault Rate",
            "Tracking Confidence",
        ],
        "Before": [
            f"{before.get('camera_reliability', {}).get('label', 'Unknown')} - {before.get('camera_reliability', {}).get('score', 0)}/100",
            f"{before['max_pelvic_drop']:.1f}°",
            f"{before['avg_pelvic_drop']:.1f}°",
            f"{before['max_knee_flexion']:.1f}°",
            f"{before['avg_knee_flexion']:.1f}°",
            f"{before['max_trunk_lean']:.1f}°",
            f"{before['avg_trunk_lean']:.1f}°",
            f"{before['max_shoulder_tilt']:.1f}°",
            f"{before['avg_shoulder_tilt']:.1f}°",
            f"{before['valgus_rate']:.1f}%",
            f"{before['movement_fault_rate']:.1f}%",
            f"{before['tracking_confidence_rate']:.1f}%",
        ],
        "After": [
            f"{after.get('camera_reliability', {}).get('label', 'Unknown')} - {after.get('camera_reliability', {}).get('score', 0)}/100",
            f"{after['max_pelvic_drop']:.1f}°",
            f"{after['avg_pelvic_drop']:.1f}°",
            f"{after['max_knee_flexion']:.1f}°",
            f"{after['avg_knee_flexion']:.1f}°",
            f"{after['max_trunk_lean']:.1f}°",
            f"{after['avg_trunk_lean']:.1f}°",
            f"{after['max_shoulder_tilt']:.1f}°",
            f"{after['avg_shoulder_tilt']:.1f}°",
            f"{after['valgus_rate']:.1f}%",
            f"{after['movement_fault_rate']:.1f}%",
            f"{after['tracking_confidence_rate']:.1f}%",
        ],
    })

    st.dataframe(comparison, use_container_width=True)

    st.subheader("Comparison Charts")

    compare_df = pd.DataFrame({
        "Before Pelvic Drop": before["pelvic_history"],
        "After Pelvic Drop": after["pelvic_history"],
        "Before Knee Flexion": before["knee_flexion_history"],
        "After Knee Flexion": after["knee_flexion_history"],
        "Before Trunk Lean": before["trunk_lean_history"],
        "After Trunk Lean": after["trunk_lean_history"],
    })

    st.line_chart(compare_df)

    st.subheader("AI Comparison Summary")

    fault_change = after["movement_fault_rate"] - before["movement_fault_rate"]
    valgus_change = after["valgus_rate"] - before["valgus_rate"]
    trunk_change = after["max_trunk_lean"] - before["max_trunk_lean"]
    pelvic_change = after["max_pelvic_drop"] - before["max_pelvic_drop"]

    if fault_change < 0:
        st.success(f"Overall movement fault rate improved by {abs(fault_change):.1f}%.")
    elif fault_change > 0:
        st.warning(f"Overall movement fault rate worsened by {fault_change:.1f}%.")
    else:
        st.info("Overall movement fault rate stayed the same.")

    if valgus_change < 0:
        st.success(f"Knee valgus risk improved by {abs(valgus_change):.1f}%.")
    elif valgus_change > 0:
        st.warning(f"Knee valgus risk worsened by {valgus_change:.1f}%.")
    else:
        st.info("Knee valgus risk stayed the same.")

    if trunk_change < 0:
        st.success(f"Trunk lean improved by {abs(trunk_change):.1f}°.")
    elif trunk_change > 0:
        st.warning(f"Trunk lean worsened by {trunk_change:.1f}°.")
    else:
        st.info("Trunk lean stayed the same.")

    if pelvic_change < 0:
        st.success(f"Pelvic drop improved by {abs(pelvic_change):.1f}°.")
    elif pelvic_change > 0:
        st.warning(f"Pelvic drop worsened by {pelvic_change:.1f}°.")
    else:
        st.info("Pelvic drop stayed the same.")


# -------------------------------------------------
# MAIN UI
# -------------------------------------------------
analysis_type = st.radio(
    "Choose Analysis Type",
    [
        "Single Video Analysis",
        "Before / After Comparison",
    ],
    horizontal=True,
)

movement_test = st.selectbox(
    "Choose Movement Test",
    [
        "Squat Analysis",
        "Running / Gait Analysis",
        "Jump Landing",
        "Posture Screen",
    ],
)

st.caption(MOVEMENT_TESTS[movement_test]["description"])

with st.expander("What this test measures"):
    for metric in MOVEMENT_TESTS[movement_test]["primary_metrics"]:
        st.write(f"• {metric}")

camera_view = st.selectbox(
    "Choose Camera View",
    [
        "Front View",
        "Side View",
        "Rear View",
        "Diagonal / Unknown",
    ],
)

st.caption(f"Camera View Guidance: {CAMERA_VIEWS[camera_view]['description']}")

with st.expander("Camera setup guidance"):
    st.write("**This view is best for:**")
    for item in CAMERA_VIEWS[camera_view]["best_for"]:
        st.write(f"• {item}")

    st.write("**This view is weak for:**")
    for item in CAMERA_VIEWS[camera_view]["weak_for"]:
        st.write(f"• {item}")

    st.write("**Filming rules for better results:**")
    st.write("• Keep the full body visible from head to feet.")
    st.write("• Keep the camera still.")
    st.write("• Avoid diagonal angles unless doing a rough screen.")
    st.write("• Use good lighting.")
    st.write("• Keep the athlete centered in the frame.")
    st.write("• For squat depth, use side view.")
    st.write("• For knee valgus, use front view.")
    st.write("• For gait pelvic drop, use rear or front view.")


if analysis_type == "Single Video Analysis":
    uploaded_video = st.file_uploader(
        "Upload Movement Video",
        type=["mp4", "mov", "avi"],
    )

    if uploaded_video is not None:
        report = analyze_video(
            uploaded_video,
            movement_test=movement_test,
            camera_view=camera_view,
            label="Single Video Report",
            client_profile=client_profile,
        )

        if report:
            show_report(report)

else:
    col1, col2 = st.columns(2)

    with col1:
        before_video = st.file_uploader(
            "Upload BEFORE Video",
            type=["mp4", "mov", "avi"],
            key="before",
        )

    with col2:
        after_video = st.file_uploader(
            "Upload AFTER Video",
            type=["mp4", "mov", "avi"],
            key="after",
        )

    if before_video is not None and after_video is not None:
        st.info("Processing BEFORE video...")

        before_report = analyze_video(
            before_video,
            movement_test=movement_test,
            camera_view=camera_view,
            label="Before Report",
            client_profile=client_profile,
        )

        st.info("Processing AFTER video...")

        after_report = analyze_video(
            after_video,
            movement_test=movement_test,
            camera_view=camera_view,
            label="After Report",
            client_profile=client_profile,
        )

        if before_report and after_report:
            show_report(before_report)
            show_report(after_report)
            compare_reports(before_report, after_report)


show_history_dashboard()
