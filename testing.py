import cv2

# Capture video from the default webcam (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    # Read each frame from the webcam
    success, frame = cap.read()

    # Check if the frame was successfully captured
    if not success:
        print("Failed to capture image")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Press 'q' to exit the webcam feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
