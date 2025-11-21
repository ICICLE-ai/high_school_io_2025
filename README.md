# YOLO Gesture Detection Project

A complete pipeline for training and deploying a YOLO-based gesture detection model. This project uses Google Gemini API to generate synthetic training data, trains a YOLO model using Ultralytics, and provides real-time gesture detection via webcam.

## Features

- **Synthetic Data Generation**: Uses Google Gemini 2.5 Flash Image API to generate diverse training images with bounding boxes
- **YOLO Training**: Automated training pipeline with train/val/test splits
- **Real-time Detection**: Live webcam gesture detection with interactive controls
- **5-Class Gesture Recognition**: Detects thumb_up, thumb_down, index_up, index_down, and rotate (stop) gestures

## Project Structure

```
high_school_io_2025/
├── test_gemini_image.py      # Generate synthetic dataset using Gemini API
├── train_yolo.py             # Train YOLO model with automatic data splitting
├── realtime_detection.py     # Real-time gesture detection via webcam
├── dataset.yaml              # YOLO dataset configuration
├── test_data/                # Generated images and labels
│   ├── images/
│   └── labels/
├── yolo_dataset/             # Organized train/val/test splits
│   ├── train/
│   ├── val/
│   └── test/
└── runs/detect/              # Training outputs and model weights
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Google Gemini API Key
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Step 1: Generate Synthetic Dataset

Generate training images using Google Gemini API:

```bash
python test_gemini_image.py
```

**What it does:**
- Generates synthetic images for 5 gesture classes using Gemini API
- Creates tight bounding boxes around hands in each image
- Saves images and YOLO-format labels to `test_data/`
- Default: 10 samples per class (configurable in script)

**Classes Generated:**
- `thumb_up` (class_id: 0) - Thumbs up gesture
- `thumb_down` (class_id: 1) - Thumbs down gesture
- `index_up` (class_id: 2) - Index finger pointing up
- `index_down` (class_id: 3) - Index finger pointing down
- `rotate` (class_id: 4) - Stop hand sign (palm facing forward)

**Output Structure:**
```
test_data/
├── images/
│   ├── thumb_up_001.jpg
│   ├── thumb_up_002.jpg
│   ├── thumb_down_001.jpg
│   └── ...
└── labels/
    ├── thumb_up_001.txt
    ├── thumb_up_002.txt
    ├── thumb_down_001.txt
    └── ...
```

**Customizing Generation:**
Edit `test_gemini_image.py` to modify:
- Number of samples per class (default: 10)
- Class configuration in `dataset_config` dictionary
- Prompt variations for diversity

### Step 2: Train YOLO Model

Train the YOLO model with automatic data splitting:

```bash
python train_yolo.py
```

**What it does:**
- Automatically splits data into train/val/test (default: 80/10/10)
- Creates YOLO dataset structure in `yolo_dataset/`
- Generates `dataset.yaml` configuration file
- Trains YOLO model (default: YOLOv8n nano model)
- Auto-detects device (MPS for Mac M1/M2, CPU otherwise)

**Training Configuration:**
Edit `train_yolo.py` to customize:
- `MODEL_SIZE`: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
- `EPOCHS`: Number of training epochs (default: 100)
- `BATCH_SIZE`: Batch size (default: 16)
- `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`: Data split ratios
- `CLASS_NAMES`: Class names (must match dataset.yaml)

**Output:**
- Best model: `runs/detect/yolo_training/weights/best.pt`
- Last model: `runs/detect/yolo_training/weights/last.pt`
- Training plots and metrics in `runs/detect/yolo_training/`

### Step 3: Real-time Detection

Run real-time gesture detection using your webcam:

```bash
python realtime_detection.py
```

**Features:**
- Live webcam feed with bounding boxes and labels
- Interactive confidence threshold adjustment
- FPS display and detection count
- Color-coded classes:
  - Green: thumb_up
  - Red: thumb_down
  - Blue: index_up
  - Orange: index_down
  - Purple: rotate

**Controls:**
- `q`: Quit
- `c`: Cycle confidence threshold (0.1, 0.15, 0.25, 0.5)
- `+` or `=`: Increase confidence by 0.05
- `-`: Decrease confidence by 0.05
- `1`: Set threshold to 0.6
- `2`: Set threshold to 0.7
- `3`: Set threshold to 0.8
- `4`: Set threshold to 0.9

**Configuration:**
Edit `realtime_detection.py` to customize:
- `MODEL_PATH`: Path to model weights (auto-detects if None)
- `CONF_THRESHOLD`: Default confidence threshold (default: 0.6)
- `CAMERA_ID`: Camera device ID (default: 0)

## Class Mapping

The project uses 5 gesture classes with the following IDs:

| Class Name | Class ID | Description |
|------------|----------|-------------|
| `thumb_up` | 0 | Thumbs up gesture |
| `thumb_down` | 1 | Thumbs down gesture |
| `index_up` | 2 | Index finger pointing up |
| `index_down` | 3 | Index finger pointing down |
| `rotate` | 4 | Stop hand sign (palm facing forward) |

## Label Format

Each label file (`.txt`) contains YOLO format annotations:

```
class_id center_x center_y width height
```

All values are normalized (0-1). Example:
```
0 0.523456 0.456789 0.234567 0.345678
```

## Dataset Configuration

The `dataset.yaml` file defines the dataset structure:

```yaml
path: /path/to/yolo_dataset
train: train/images
val: val/images
test: test/images

nc: 5
names: ['thumb_up', 'thumb_down', 'index_up', 'index_down', 'rotate']
```

## Requirements

- Python 3.8+
- PyTorch (with MPS support for Mac M1/M2)
- Ultralytics YOLO
- OpenCV
- Google Gemini API key

See `requirements.txt` for complete dependency list.

## Notes

- **API Key**: The Gemini API key is required for data generation. Get it from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Model Weights**: Pre-trained YOLOv8 weights are automatically downloaded on first run
- **Device Detection**: The training script automatically uses MPS (Metal Performance Shaders) on Mac M1/M2 for GPU acceleration
- **Data Continuity**: The data generation script continues from existing indices, so you can add more samples without overwriting existing data
- **Confidence Threshold**: Higher thresholds (0.7-0.9) reduce false positives but may miss some detections

## Troubleshooting

**Camera not opening:**
- Try changing `CAMERA_ID` to 1 or check camera permissions
- On Mac, grant camera permissions in System Settings

**Model not found:**
- Ensure you've trained a model first using `train_yolo.py`
- Or specify `MODEL_PATH` in `realtime_detection.py`

**API errors:**
- Verify your `GOOGLE_API_KEY` is set correctly in `.env`
- Check API quota and rate limits
- The script includes automatic retry logic

**Training issues:**
- Reduce `BATCH_SIZE` if you run out of memory
- Use smaller model size ('n' instead of 's', 'm', etc.)
- Ensure you have enough training data (recommended: 100+ images per class)
