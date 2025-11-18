# Synthetic Image Data Generation for YOLO Training

This project can build a YOLO-ready dataset in two ways:

1. **Local augmentation mode** – OpenCV/Pillow pipeline that perturbs your seed images.
2. **Google Gemini image mode** – Sends your seed images to Google's `gemini-2.5-flash-image` model, asks it for tightly cropped samples, and stores both the returned JPEGs and YOLO labels. All API calls include retry + timeout protection.

## Features

- **Local augmentation pipeline** (OpenCV + Pillow) for quick synthetic variants
- **Gemini image API integration** with retry/backoff + timeout safeguards
- **Class-specific variant plan** (default matches go_up/go_down/rotate seeds)
- **Tight bounding boxes** from API responses or automatic contour extraction
- **YOLO labels** written in Ultralytics-compatible format
- **Deterministic folder structure** (`images/` + `labels/`)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local augmentation (default)

1. **Prepare input images**
   - Create an `input_images` folder.
   - Drop one seed image per class (file stem should match the class name).

2. **Run the script**
```bash
python generate_synthetic_data.py
```

3. **Customize options**
```bash
python generate_synthetic_data.py --input-dir input_images --output-dir test_data --num-samples 50
```

### Google Gemini image mode

1. **Seed images**
   - Place the following images in `input_images/`:
     - `go_up_1.(jpg|png)`
     - `go_up_2.(jpg|png)`
     - `go_down_1.(jpg|png)`
     - `go_down_2.(jpg|png)`
     - `rotate.(jpg|png)`

2. **API key**
   - Export your key: `export GOOGLE_API_KEY=your-key`
   - The script now auto-loads `.env` (via `python-dotenv`). Make sure it contains `GOOGLE_API_KEY=...`.
   - Requests send the key via the `x-goog-api-key` header (required by Gemini image endpoints).

3. **Optional plan JSON**
   - Use `--variant-plan path/to/plan.json` to override the default counts (`go_up`: 25 from each seed, `go_down`: same, `rotate`: 50 total).

4. **Run Gemini image mode**
```bash
python generate_synthetic_data.py \
  --use-nano-banana \
  --input-dir input_images \
  --output-dir test_data \
  --nano-timeout 45 \
  --nano-retries 5
```

## Output Structure

Every run creates the following layout:

```
test_data/
├── images/
│   ├── go_up_1_000.jpg
│   ├── go_up_2_001.jpg
│   ├── ...
│   ├── go_down_000.jpg
│   └── ...
└── labels/
    ├── go_up_1_000.txt
    ├── go_up_2_001.txt
    ├── ...
```

## Label Format

Each `.txt` file contains YOLO format annotations:
```
class_id center_x center_y width height
```

All values are normalized (0-1). Example:
```
0 0.523456 0.456789 0.234567 0.345678
```

## Class Mapping (default plan)

- `go_up` → class_id: 0
- `go_down` → class_id: 1
- `rotate` → class_id: 2

## Variant plan JSON (optional)

Override the default `go_up/go_down/rotate` counts by creating a JSON file:
```json
{
  "go_up": [
    {"seed": "go_up_1", "count": 25},
    {"seed": "go_up_2", "count": 25}
  ],
  "go_down": [
    {"seed": "go_down_1", "count": 25},
    {"seed": "go_down_2", "count": 25}
  ],
  "rotate": [
    {"seed": "rotate", "count": 50}
  ]
}
```
Pass the file via `--variant-plan path/to/file.json`.

## Notes

- Gemini image mode requires `GOOGLE_API_KEY` and honours `--nano-timeout`/`--nano-retries`.
- Local mode automatically extracts tight boxes when the API is not used.
- Images are saved as `.jpg`; labels are YOLO txt files placed in `labels/`.
- Run `python generate_synthetic_data.py --help` to see every flag.

