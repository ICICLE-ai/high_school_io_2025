#!/usr/bin/env python3
"""
Real-time gesture detection using trained YOLO model
Opens Mac camera and performs live detection with bounding boxes and labels
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO

from gesture_config import CLASS_COLORS, CLASS_NAMES


def load_model(model_path=None):
    """
    Load the trained YOLO model
    
    Args:
        model_path: Path to model weights. If None, tries to find best.pt automatically
    
    Returns:
        Loaded YOLO model
    """
    if model_path is None:
        # Try to find the best model from training (prioritize yolo_training2)
        possible_paths = [
            "runs/detect/yolo_training3/weights/best.pt",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                model_path = path
                print(f"✅ Found model: {model_path}")
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "Could not find trained model. Please specify model_path or train a model first.\n"
                "Expected paths: runs/detect/yolo_training2/weights/best.pt"
            )
    
    print(f"📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    return model


def get_class_colors():
    """Return color mapping for each class"""
    return CLASS_COLORS


def draw_detections(frame, results, class_names, conf_threshold=0.25):
    """
    Draw bounding boxes and labels on frame using YOLO's built-in annotation
    
    Args:
        frame: Input frame (numpy array)
        results: YOLO detection results
        class_names: List of class names
        conf_threshold: Confidence threshold for display
    
    Returns:
        Frame with drawn detections
    """
    if not results or len(results) == 0:
        return frame

    annotated_frame = frame.copy()
    colors = get_class_colors()

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated_frame

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        if confidence < conf_threshold:
            continue

        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        color = colors.get(class_id, (255, 255, 255))

        cv2.rectangle(
            annotated_frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

        label = f"{class_name}: {confidence:.2f}"
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        label_y = max(int(y1) - 10, 30)

        cv2.rectangle(
            annotated_frame,
            (int(x1), label_y - text_height - 8),
            (int(x1) + text_width + 10, label_y + 5),
            color,
            -1
        )

        cv2.putText(
            annotated_frame,
            label,
            (int(x1) + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return annotated_frame


def main():
    """Main real-time detection loop"""
    # Configuration
    MODEL_PATH = None  # Auto-detect (prioritizes yolo_training2), or specify: "runs/detect/yolo_training2/weights/best.pt"
    CONF_THRESHOLD = 0.1  # Confidence threshold (0.0 to 1.0) - Lower default to catch more detections
    CAMERA_ID = 0  # Usually 0 for default camera, try 1 if 0 doesn't work
    WINDOW_NAME = "Real-time Gesture Detection"
    
    print("=" * 60)
    print("Real-time YOLO Gesture Detection")
    print("=" * 60)
    
    # Load model
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Auto-detect device
    try:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
            print("🖥️  Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
        else:
            device = "cpu"
            print("🖥️  Using CPU")
    except Exception:
        device = "cpu"
        print("🖥️  Using CPU")
    
    # Open camera
    print(f"\n📹 Opening camera (ID: {CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {CAMERA_ID}")
        print("💡 Try changing CAMERA_ID to 1 or check camera permissions")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera opened successfully!")
    print("\n" + "=" * 60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to cycle confidence threshold (0.1, 0.2, 0.3, 0.5)")
    print("  - Press '+' or '=' to increase confidence by 0.05 (up to 0.99)")
    print("  - Press '-' to decrease confidence by 0.05 (down to 0.05)")
    print("  - Press '1' to set threshold to 0.3")
    print("  - Press '2' to set threshold to 0.5")
    print("  - Press '3' to set threshold to 0.7")
    print("  - Press '4' to set threshold to 0.9")
    print(f"  - Current confidence threshold: {CONF_THRESHOLD}")
    print("  - Check console for detection details (updated every second)")
    print("=" * 60 + "\n")
    
    # FPS calculation
    fps_frame_count = 0
    fps = 0.0
    last_fps_update = time.time()
    
    current_conf = CONF_THRESHOLD
    
    print("🎥 Starting video stream...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame from camera")
                break
            
            # Run inference using the active confidence threshold
            try:
                results = model.predict(
                    frame,
                    conf=current_conf,
                    device=device,
                    verbose=False,
                    imgsz=640
                )
            except Exception as e:
                print(f"❌ Error during inference: {e}")
                continue
            
            # Count detections for debugging
            num_detections = 0
            if results and len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            
            # Draw detections
            frame = draw_detections(frame, results, CLASS_NAMES, current_conf)
            
            # Calculate and display FPS
            fps_frame_count += 1
            current_time = time.time()
            elapsed = current_time - last_fps_update
            
            if elapsed >= 1.0:  # Update FPS every second
                fps = fps_frame_count / elapsed
                fps_frame_count = 0
                last_fps_update = current_time
            
            # Draw FPS and info on frame
            info_text = f"FPS: {fps:.1f} | Conf: {current_conf:.2f} | Detections: {num_detections} | Press 'q' to quit"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show class names legend
            legend_y = 60
            for i, class_name in enumerate(CLASS_NAMES):
                color = get_class_colors()[i]
                legend_text = f"{i}: {class_name}"
                cv2.putText(
                    frame,
                    legend_text,
                    (10, legend_y + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Show frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('c'):
                # Cycle through confidence thresholds
                if current_conf >= 0.5:
                    current_conf = 0.1
                elif current_conf >= 0.3:
                    current_conf = 0.1
                elif current_conf >= 0.2:
                    current_conf = 0.3
                elif current_conf >= 0.1:
                    current_conf = 0.2
                else:
                    current_conf = 0.5
                print(f"🔧 Confidence threshold changed to: {current_conf}")
            elif key == ord('+') or key == ord('='):
                # Increase confidence
                current_conf = min(0.99, current_conf + 0.05)
                print(f"🔧 Confidence threshold increased to: {current_conf:.2f}")
            elif key == ord('-'):
                # Decrease confidence
                current_conf = max(0.05, current_conf - 0.05)
                print(f"🔧 Confidence threshold decreased to: {current_conf:.2f}")
            elif key == ord('1'):
                # Quick set to 0.3
                current_conf = 0.3
                print(f"🔧 Confidence threshold set to: {current_conf:.2f}")
            elif key == ord('2'):
                # Quick set to 0.5
                current_conf = 0.5
                print(f"🔧 Confidence threshold set to: {current_conf:.2f}")
            elif key == ord('3'):
                # Quick set to 0.7
                current_conf = 0.7
                print(f"🔧 Confidence threshold set to: {current_conf:.2f}")
            elif key == ord('4'):
                # Quick set to 0.9
                current_conf = 0.9
                print(f"🔧 Confidence threshold set to: {current_conf:.2f}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera released and windows closed")


if __name__ == "__main__":
    main()
