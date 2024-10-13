import cv2
import mediapipe as mp
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Key, Controller
import time

key_controller = Controller()

pose_tracker = PoseDetector()

hand_detector = HandDetector(maxHands=2, detectionCon=0.7)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Unable to access webcam")
    exit()

last_turn_time = 0
cooldown_duration = 0.75 

_, initial_frame = video_capture.read()
frame_height, frame_width = initial_frame.shape[:2]
half_screen_height = frame_height // 2

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hands, frame = hand_detector.findHands(frame, draw=True, flipType=True)

    pose_results = pose.process(rgb_frame)

    cv2.line(frame, (0, half_screen_height), (frame_width, half_screen_height), (255, 255, 0), 2)

    neutral_state = True  # Start in a neutral state
    is_jumping = False
    is_sliding = False

    current_time = time.time()

    ### Crouching 
    if pose_results.pose_landmarks:
        left_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        torso_center_y = (left_hip.y + right_hip.y) * frame_height // 2

        if torso_center_y > half_screen_height and not is_jumping:
            cv2.putText(frame, "Action: Slide (S)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("Action: Slide (S)")
            key_controller.press('s')
            key_controller.release('s')
            is_sliding = True
            neutral_state = False

    ### Jumping
    if hands and len(hands) == 2 and not is_sliding:  # Both hands detected
        left_hand = hands[0]['center']
        right_hand = hands[1]['center']
        if left_hand[1] < frame_height * 0.3 and right_hand[1] < frame_height * 0.3:  # Both hands raised
            cv2.putText(frame, "Action: Jump (W)", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            print("Action: Jump (W)")
            key_controller.tap('w')
            is_jumping = True
            neutral_state = False

    ### Hoverboard

        left_hand = hands[0]['center']
        right_hand = hands[1]['center']
        if abs(left_hand[0] - right_hand[0]) < frame_width * 0.1:
             cv2.putText(frame, "Action: Hoverboard (Space)", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
             print("Action: Hoverboard)")
             key_controller.tap(Key.space)
             neutral_state = False

    ### Swiping Left/Right for Turning (cvzone HandDetector) with Cooldown ###
    if hands:
        for hand in hands:
            hand_x = hand['center'][0]  # X coordinate of hand

            # Check for cooldown
            if current_time - last_turn_time >= cooldown_duration:
                # Swipe Left
                if hand_x > frame_width * 0.7:  # Was previously on the left
                    cv2.putText(frame, "Action: Turn Left (A)", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Action: Turn Left (A)")
                    key_controller.press('a')
                    key_controller.release('a')
                    last_turn_time = current_time
                    neutral_state = False

                # Swipe Right
                elif hand_x < frame_width * 0.3:  # Was previously on the right
                    cv2.putText(frame, "Action: Turn Right (D)", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Action: Turn Right (D)")
                    key_controller.press('d')
                    key_controller.release('d')
                    last_turn_time = current_time
                    neutral_state = False

    # Neutral State- no action is detected
    if neutral_state:
        cv2.putText(frame, "Neutral State", (frame_width // 2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Interrupt program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Program interrupted by user.")
        break

    # Display modified video frame
    cv2.imshow("Gesture-Based Game Control", frame)

# Clean up and release resources
video_capture.release()
cv2.destroyAllWindows()
