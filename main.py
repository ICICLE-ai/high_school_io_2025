import cv2
from ultralytics import YOLO
from SoftwarePilot import SoftwarePilot

# Load YOLO model
model = YOLO('runs/detect/yolo_training2/weights/best.pt')

# Setup drone
sp = SoftwarePilot()
drone = sp.setup_drone("parrot_anafi", 1, "None")
drone.connect()

# Open camera
cap = cv2.VideoCapture(0)

# Action mapping
actions = {
    'go_up': lambda: drone.piloting.takeoff(),
    'go_down': lambda: drone.piloting.land(),
    'rotate': lambda: drone.piloting.move_by(0, 0, 0, 90, wait=True)
}

print("Camera opened. Show hand gestures to control drone. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Execute drone action
            if label in actions:
                print(f"Detected: {label}")
                actions[label]()

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('Drone Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
drone.disconnect()