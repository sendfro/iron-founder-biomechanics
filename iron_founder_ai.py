import cv2
import mediapipe as mp
import numpy as np

# 1. Initialize the AI Vision Engine
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 2. The Biomechanical Math Engine
def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def main():
    # Make sure this exactly matches your video file name
    video_input = 'drone_footage.mp4' 
    cap = cv2.VideoCapture(video_input)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup the output file
    out = cv2.VideoWriter('master_audit_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 3. Boot up the Pose Model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            
            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 4. The Master Anomaly Logic
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Extract Coordinates
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                
                l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate Knee Angles to determine current stance
                r_angle = calculate_angle(r_hip, r_knee, r_ankle)
                l_angle = calculate_angle(l_hip, l_knee, l_ankle)
                
                # Calculate Distances for Valgus
                knee_dist = abs(r_knee[0] - l_knee[0])
                ankle_dist = abs(r_ankle[0] - l_ankle[0])
                
                # Calculate Pelvic Tilt for Running Gait
                hip_tilt_radians = np.arctan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0])
                hip_tilt_angle = np.abs(hip_tilt_radians * 180.0 / np.pi)
                pelvic_deviation = min(hip_tilt_angle, abs(180 - hip_tilt_angle))
                
                # Default Safe State
                warning_color = (0, 255, 0) # Green
                warning_text = "SYSTEM ACTIVE: FORM SOLID"
                
                # --- STATE MACHINE LOGIC ---
                
                # STATE 1: SQUATTING (Knees are bent)
                if r_angle < 150 or l_angle < 150:
                    warning_text = "MODE: SQUAT AUDIT"
                    if knee_dist < (ankle_dist * 0.8):
                        warning_color = (0, 0, 255) # Red
                        warning_text = "WARNING: KNEE VALGUS DETECTED"
                
                # STATE 2: UPRIGHT / RUNNING (Knees are mostly straight)
                else:
                    warning_text = f"MODE: GAIT AUDIT (Tilt: {int(pelvic_deviation)} deg)"
                    # Threshold: 8 degrees of drop triggers a warning
                    if pelvic_deviation > 8.0:
                        warning_color = (0, 165, 255) # Orange/Red
                        warning_text = f"WARNING: PELVIC DROP ({int(pelvic_deviation)} DEG)"
                
                # 5. Display the HUD
                cv2.putText(image, warning_text, (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2, cv2.LINE_AA)
                
            except:
                pass
            
            # Draw the skeleton
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=warning_color, thickness=2, circle_radius=2)
            )
            
            cv2.imshow('Iron Founder Biomechanic Audit v2', image)
            out.write(image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()