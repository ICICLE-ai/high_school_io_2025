import cv2
import json
import time
import copy
from collections import defaultdict
from ultralytics import YOLO

# Switch between CV-only testing and real drone execution.
# - "test": never connects/sends commands to drone
# - "live": enables drone connection + command execution
RUN_MODE = "test"

if RUN_MODE not in {"test", "live"}:
    raise ValueError("RUN_MODE must be either 'test' or 'live'.")

if RUN_MODE == "live":
    from SoftwarePilot import SoftwarePilot

# Load actions from JSON
try:
    with open('actions.json', 'r') as f:
        actions_config = json.load(f)
except FileNotFoundError:
    print("actions.json not found! Please create it.")
    actions_config = {}

# Load YOLO model
model = YOLO('runs/detect/yolo_training3/weights/best.pt')

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
        # CAP_AVFOUNDATION is the native macOS backend.
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"Camera opened on index {idx}.")
            return cap
        cap.release()
    return None


cap = open_camera()
if cap is None:
    raise SystemExit(
        "Could not open any camera index (0-4). Check macOS camera permission "
        "for Terminal/Cursor and ensure no other app is using the camera."
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


def pick_command(action_config, execution_count):
    _ = execution_count
    return action_config


def action_limit(action_config):
    max_executions = action_config.get("max_executions")
    if max_executions is None:
        return None
    return int(max_executions)


def action_limit_text(action_name, action_config):
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

# State for debouncing
last_action_time = 0
cooldown_seconds = 2.0  # 2 seconds cooldown between gestures
action_counts = {label: 0 for label in actions_config}
subaction_counts = {label: defaultdict(int) for label in actions_config}
balance_positions = defaultdict(float)
balance_directions = defaultdict(lambda: 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, conf=0.9)

    current_time = time.time()

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            # Execute configured action with limits and sequence support.
            if label in actions_config and current_time - last_action_time > cooldown_seconds:
                action_config = actions_config[label]
                max_executions = action_limit(action_config)
                already_executed = action_counts[label]

                if max_executions is None or already_executed < max_executions:
                    command_to_run = pick_command(action_config, already_executed)
                    command_to_run, balance_group, balance_step = resolve_balanced_command(
                        command_to_run,
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
                            subaction_name = command_to_run.get("name")
                            if subaction_name:
                                subaction_counts[label][subaction_name] += 1
                            print(f"Detected: {label}. Execution #{action_counts[label]}")
                            last_action_time = current_time

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display frame
    y_offset = 25
    for action_name, action_config in actions_config.items():
        bounded_progress = bounded_progress_for_display(action_config, balance_positions)
        if bounded_progress is not None:
            executed, bounded_limit = bounded_progress
            limit_text = str(bounded_limit)
        else:
            executed = action_counts.get(action_name, 0)
            limit_text = action_limit_text(action_name, action_config)
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
        down_left = max(0, v_pos - 0)
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
        backward_left = max(0, fb_pos - (-2))
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