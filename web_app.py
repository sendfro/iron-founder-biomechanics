import sys
import os
import streamlit as st
import mediapipe as mp
import numpy as np
import tempfile
import cv2

# --- UI Setup ---
st.set_page_config(page_title="Iron Founder Biomechanics", layout="wide")
st.title("Iron Founder AI: Motion Capture Engine")
st.markdown("Upload your video below for instant biomechanical analysis.")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

uploaded_file = st.file_uploader("Upload Phone or Drone Video", type=['mp4', 'mov'])

if uploaded_file is not None:
    st.success("Video uploaded successfully. Initializing AI Engine...")
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Get total frames for our progress bar
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # UI Elements for processing
    progress_text = st.empty()
    progress_bar = st.progress(0)
    stframe = st.empty()
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # --- REPORT TRACKING VARIABLES ---
    current_frame = 0
    valgus_error_count = 0
    hip_drop_error_count = 0
    max_hip_drop = 0.0
    
    # Array to store data for our chart
    pelvic_drop_history = []
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            current_frame += 1
            
            # Update Progress Bar
            if total_video_frames > 0:
                progress = min(current_frame / total_video_frames, 1.0)
                progress_bar.progress(progress)
                progress_text.text(f"Processing Frame {current_frame} of {total_video_frames}...")

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            
            # Scoped variables declared cleanly before the logic block
            warning_color = (0, 255, 0)
            warning_text = "SYSTEM ACTIVE: FORM SOLID"
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                
                # FIXED: Euclidean 2D distance calculation for correct spatial awareness
                knee_dist = np.linalg.norm(np.array(r_knee) - np.array(l_knee))
                ankle_dist = np.linalg.norm(np.array(r_ankle) - np.array(l_ankle))
                
                hip_tilt_radians = np.arctan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0])
                hip_tilt_angle = np.abs(hip_tilt_radians * 180.0 / np.pi)
                pelvic_deviation = min(hip_tilt_angle, abs(180 - hip_tilt_angle))
                
                # Record the angle for the graph
                pelvic_drop_history.append(pelvic_deviation)
                
                if pelvic_deviation > max_hip_drop:
                    max_hip_drop = pelvic_deviation
                
                if pelvic_deviation > 8.0:
                    hip_drop_error_count += 1
                
                is_squatting = r_angle < 150 or l_angle < 150
                if is_squatting and (knee_dist < (ankle_dist * 0.8)):
                    valgus_error_count += 1

                if is_squatting:
                    warning_text = "MODE: SQUAT AUDIT"
                    if knee_dist < (ankle_dist * 0.8):
                        warning_color = (255, 0, 0) 
                        warning_text = "WARNING: KNEE VALGUS DETECTED"
                else:
                    warning_text = f"MODE: GAIT AUDIT (Tilt: {int(pelvic_deviation)} deg)"
                    if pelvic_deviation > 8.0:
                        warning_color = (255, 165, 0) 
                        warning_text = f"WARNING: PELVIC DROP ({int(pelvic_deviation)} DEG)"
                
                cv2.putText(image, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2, cv2.LINE_AA)
                
            except:
                # If no landmarks found, still record a 0 so the graph timeline doesn't break
                pelvic_drop_history.append(0)
                pass
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=warning_color, thickness=2, circle_radius=2)
                )
            
            stframe.image(image, channels="RGB", use_container_width=True)

    cap.release()
    
    # FIXED: Clean up the temporary file from the OS to prevent memory leaks
    try:
        os.remove(tfile.name)
    except Exception as e:
        st.error(f"Error removing temp file: {e}")
    
    # Clear the processing UI
    stframe.empty() 
    progress_text.empty()
    progress_bar.empty()
    
    # --- THE DIAGNOSTIC DASHBOARD ---
    st.markdown("---")
    st.header("📊 Final Biomechanical Report")
    
    if current_frame > 0:
        gait_instability_rate = (hip_drop_error_count / current_frame) * 100
        valgus_rate = (valgus_error_count / current_frame) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Max Pelvic Drop", f"{int(max_hip_drop)}°", "Should be < 8°", delta_color="inverse")
        col2.metric("Gait Instability Rate", f"{int(gait_instability_rate)}%", "Frames w/ Hip Drop", delta_color="inverse")
        col3.metric("Knee Valgus Risk", f"{int(valgus_rate)}%", "Frames w/ Valgus", delta_color="inverse")
        
        st.markdown("---")
        
        # Render the Data Visualization Chart
        st.subheader("📈 Kinematic Data: Pelvic Drop Over Time")
        st.caption("This chart displays your pelvic tilt angle across the duration of the movement. Values spiking above the 8-degree threshold indicate instability.")
        st.line_chart(pelvic_drop_history)
        
        st.markdown("---")
        st.subheader("Engine Summary")
        if max_hip_drop > 8.0:
            st.warning("⚠️ **Gait Warning:** The AI detected significant pelvic drop during your run. Your Gluteus Medius may be fatiguing, causing your opposite hip to drop. This places lateral stress on the IT band and groin.")
        else:
            st.success("✅ **Gait Clear:** Pelvic stability maintained throughout the entire movement.")
            
        if valgus_rate > 0:
            st.error("🚨 **Squat Warning:** Knee valgus (inward caving) detected. Ensure your knees track perfectly over your toes during deep flexion.")
            
    else:
        st.error("No movement detected to analyze.")
