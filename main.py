import cv2
import json
import os
import sys
import time
import copy
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dynamic_gestures"))
from utils import targets  # shared gesture vocabulary — used by both detectors

# Switch between CV-only testing and real drone execution.
# - "test": never connects/sends commands to drone
# - "live": enables drone connection + command execution
RUN_MODE = "test"

if RUN_MODE not in {"test", "live"}:
    raise ValueError("RUN_MODE must be either 'test' or 'live'.")

if RUN_MODE == "live":
    from SoftwarePilot import SoftwarePilot

# Load actions from JSON
_actions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'actions.json')
try:
    with open(_actions_path, 'r') as f:
        actions_config = json.load(f)
except FileNotFoundError:
    print(f"actions.json not found at {_actions_path}! Please create it.")
    actions_config = {}

# ── DETECTOR GATEWAY ──────────────────────────────────────────────────────────
# Switch DETECTOR to choose the gesture recognition engine.
#
#   "dynamic_gestures"  — built-in pre-trained ONNX models, no setup needed.
#   "yolo"              — your own trained YOLO model (.pt file).
#
# To use YOLO:
#   1. Set DETECTOR = "yolo"
#   2. Set YOLO_MODEL_PATH to your .pt weights file
#   3. Fill YOLO_CLASS_MAP: {yolo_class_id (int): "gesture_label" (str)}
#      Labels must match keys in actions.json and appear in the targets list.
DETECTOR = "dynamic_gestures"

YOLO_MODEL_PATH = "runs/detect/yolo_training/weights/best.pt"
YOLO_CONF = 0.3
YOLO_CLASS_MAP = {
    # example — edit to match your trained model's class IDs
    0: "like",
    1: "dislike",
    2: "stop",
    3: "peace",
}
# ─────────────────────────────────────────────────────────────────────────────

if DETECTOR == "dynamic_gestures":
    from main_controller import MainController
    _dg_dir = os.path.join(os.path.dirname(__file__), "dynamic_gestures")
    _controller = MainController(
        os.path.join(_dg_dir, "models", "hand_detector.onnx"),
        os.path.join(_dg_dir, "models", "crops_classifier.onnx"),
    )

    def detect(frame):
        return _controller(frame)

elif DETECTOR == "yolo":
    from ultralytics import YOLO as _YOLO
    _yolo_model = _YOLO(YOLO_MODEL_PATH)
    _label_to_idx = {label: i for i, label in enumerate(targets)}

    def detect(frame):
        results = _yolo_model(frame, conf=YOLO_CONF, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), [], []
        bboxes = boxes.xyxy.cpu().numpy()
        ids = [None] * len(bboxes)
        label_indices = [
            _label_to_idx.get(YOLO_CLASS_MAP.get(int(cls)))
            for cls in boxes.cls.cpu().numpy()
        ]
        return bboxes, ids, label_indices

else:
    raise ValueError(f"Unknown DETECTOR: {DETECTOR!r}. Use 'dynamic_gestures' or 'yolo'.")

# Setup drone
drone = None
drone_connected = False
if RUN_MODE == "live":
    sp = SoftwarePilot()
    drone = sp.setup_drone("parrot_anafi", 1, "None")
    try:
        drone.connect()
        drone_connected = True
        print("Drone connected.")
    except Exception as e:
        print(f"Failed to connect to drone. Running CV-only fallback: {e}")
else:
    print("RUN_MODE=test: drone control disabled.")

# Open camera
def open_camera(camera_indices=(0, 1, 2, 3, 4)):
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened on index {idx}.")
            return cap
        cap.release()
    return None


cap = open_camera()
if cap is None:
    raise SystemExit(
        "Could not open any camera index (0-4). Ensure the camera is connected "
        "and not in use by another application."
    )

def execute_action(drone, command_config):
    component_name = command_config.get("component")
    action_name = command_config.get("action")
    args = command_config.get("args", [])
    kwargs = command_config.get("kwargs", {})

    try:
        if component_name and hasattr(drone, component_name):
            component = getattr(drone, component_name)
            if hasattr(component, action_name):
                func = getattr(component, action_name)
                func(*args, **kwargs)
                return True
            print(f"Action {action_name} not found on {component_name}")
            return False
        print(f"Component {component_name} not found on drone")
        return False
    except Exception as e:
        print(f"Error executing {action_name}: {e}")
        return False


def action_limit(action_config):
    max_executions = action_config.get("max_executions")
    if max_executions is None:
        return None
    return int(max_executions)


def action_limit_text(action_config):
    max_executions = action_limit(action_config)
    if max_executions is not None:
        return str(max_executions)

    balance = action_config.get("balance")
    if isinstance(balance, dict):
        min_pos = float(balance.get("min", 0))
        max_pos = float(balance.get("max", 0))
        # For bounded dynamic controls:
        # - vertical [0,2]  -> limit 2
        # - forward_back [-2,2] -> limit 2 per direction
        bounded_limit = int(max(abs(min_pos), abs(max_pos)))
        return f"2" if bounded_limit == 2 else f"bounded:{bounded_limit}"

    return "unlimited"


def bounded_progress_for_display(action_config, balance_positions):
    balance = action_config.get("balance")
    if not isinstance(balance, dict):
        return None

    group = balance.get("group")
    delta = float(balance.get("delta", 0))
    min_pos = float(balance.get("min", 0))
    max_pos = float(balance.get("max", 0))
    if not group:
        return None

    pos = float(balance_positions[group])
    progress = None
    limit = int(max(abs(min_pos), abs(max_pos)))

    # Vertical plus/minus display (range 0..2).
    if group == "vertical":
        if delta > 0:
            progress = pos
        else:
            progress = max_pos - pos
        limit = int(max_pos - min_pos)

    # Forward/backward plus/minus display (range -2..2).
    elif group == "forward_back":
        if delta > 0:
            progress = pos if pos >= 0 else max_pos + pos
        else:
            progress = -pos if pos <= 0 else max_pos - pos
        limit = int(max_pos)

    if progress is None:
        return None
    progress = int(max(0, min(limit, progress)))
    return progress, max(1, limit)


def resolve_balanced_command(action_config, balance_positions, balance_directions):
    balance = action_config.get("balance")
    if not isinstance(balance, dict):
        return action_config, None, 0

    group = balance.get("group")
    delta = float(balance.get("delta", 0))
    min_pos = float(balance.get("min", -float("inf")))
    max_pos = float(balance.get("max", float("inf")))
    arg_index = int(balance.get("arg_index", 0))
    auto_reverse = bool(balance.get("auto_reverse", False))
    if not group or delta == 0:
        return None, None, 0

    current_pos = balance_positions[group]
    direction = balance_directions[group]
    step = delta

    if auto_reverse:
        if current_pos >= max_pos and direction > 0:
            direction = -1
        elif current_pos <= min_pos and direction < 0:
            direction = 1
        balance_directions[group] = direction
        step = abs(delta) * direction

    new_pos = current_pos + step
    if new_pos < min_pos or new_pos > max_pos:
        return None, group, step

    command = copy.deepcopy(action_config)
    args = list(command.get("args", []))
    if arg_index < 0 or arg_index >= len(args):
        return None, group, step
    args[arg_index] = step
    command["args"] = args
    return command, group, step

print(f"Detection started in {RUN_MODE} mode. Press 'q' to quit.")
cv2.namedWindow('CV Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('CV Detection', 1280, 720)

# State for debouncing
last_action_time = 0
cooldown_seconds = 2.0  # 2 seconds cooldown between gestures
action_counts = {label: 0 for label in actions_config}
balance_positions = defaultdict(float)
balance_directions = defaultdict(lambda: 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    current_time = time.time()
    bboxes, ids, labels = detect(frame)

    # Process detections
    if bboxes is not None and bboxes.shape[0] > 0:
        bboxes_int = bboxes.astype(np.int32)
        for i in range(bboxes_int.shape[0]):
            box = bboxes_int[i]
            label_idx = labels[i]
            label = targets[label_idx] if label_idx is not None else None

            if label is not None and label in actions_config and current_time - last_action_time > cooldown_seconds:
                cfg = actions_config[label]
                max_executions = action_limit(cfg)
                already_executed = action_counts[label]

                if max_executions is None or already_executed < max_executions:
                    command_to_run, balance_group, balance_step = resolve_balanced_command(
                        cfg,
                        balance_positions,
                        balance_directions
                    )
                    if command_to_run is not None:
                        if RUN_MODE == "live" and drone_connected:
                            success = execute_action(drone, command_to_run)
                        else:
                            success = True

                        if success:
                            action_counts[label] += 1
                            if balance_group is not None:
                                balance_positions[balance_group] += balance_step
                            print(f"Detected: {label}. Execution #{action_counts[label]}")
                            last_action_time = current_time

            # Draw bounding box only for gestures configured in actions.json
            if label and label in actions_config:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    y_offset = 25
    for action_name, action_cfg in actions_config.items():
        bounded_progress = bounded_progress_for_display(action_cfg, balance_positions)
        if bounded_progress is not None:
            executed, bounded_limit = bounded_progress
            limit_text = str(bounded_limit)
        else:
            executed = action_counts.get(action_name, 0)
            limit_text = action_limit_text(action_cfg)
        status_text = f"{action_name}: {executed}/{limit_text}"
        cv2.putText(
            frame,
            status_text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )
        y_offset += 24

    # Display dynamic bounded movement state.
    if "vertical" in balance_positions:
        v_pos = int(balance_positions["vertical"])
        up_left = max(0, 2 - v_pos)
        down_left = max(0, v_pos)
        cv2.putText(
            frame,
            f"vertical position={v_pos} | up_left={up_left} down_left={down_left}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 120),
            1
        )
        y_offset += 20

    if "forward_back" in balance_positions:
        fb_pos = int(balance_positions["forward_back"])
        forward_left = max(0, 2 - fb_pos)
        backward_left = max(0, fb_pos + 2)
        cv2.putText(
            frame,
            f"forward_back position={fb_pos} | forward_left={forward_left} backward_left={backward_left}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (120, 255, 220),
            1
        )
        y_offset += 20

    cv2.imshow('CV Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if drone_connected:
    try:
        drone.disconnect()
    except Exception:
        pass