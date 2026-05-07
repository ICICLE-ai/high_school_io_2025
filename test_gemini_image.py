#!/usr/bin/env python3
"""
Generate synthetic dataset using Gemini image generation API.
Creates YOLO-format dataset with images and labels.

IMPORTANT: Bounding boxes:
- Should NOT be drawn on the image - images should be clean without annotations
- Bounding box coordinates must be provided in JSON format only
- Coordinates must define perfect rectangles (straight horizontal/vertical lines, 90-degree corners)
- Coordinates must be tightly fitted around the hand with minimal padding
- Contain ONLY ONE hand/object per bounding box (no multiple objects combined)
- When used to draw a bounding box, should create a tight rectangular box around the hand

BACKGROUND REQUIREMENTS:
- Use diverse backgrounds and positions(indoor, outdoor, neutral, various lighting)
- Ensure clear visual separation between hand and background
- Ensure the hand is in different positions and orientations in different images
- Ensure the hand is in different backgrounds and positions in different images
- Ensure the hand is in different lighting and positions in different images
- Ensure the hand is in different textures and positions in different images
- Ensure the hand is in different shapes and positions in different images
- Ensure the hand is in different sizes and positions in different images
- Ensure the hand is in different colors and positions in different images
"""

import base64
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from gesture_config import CLASS_NAMES, CLASS_TO_ID


def build_prompt(class_name: str, sample_index: int) -> str:
    """Build the prompt for image generation."""
    # Variation hints for diversity
    variation_hints = [
        "vary the hand angle and position",
        "use different lighting and background",
        "change the hand orientation and perspective",
        "vary the skin tone and hand size",
        "use different camera angles and distances",
    ]
    variation = variation_hints[sample_index % len(variation_hints)]
    
    # Gesture descriptions
    gesture_descriptions = {
        "thumb_up": "thumbs up gesture",
        "thumb_down": "thumbs down gesture",
        "rotate": "stop hand sign with the palm facing forward",
        "peace_sign": "raised index and middle fingers in a V sign gesture",
    }
    
    gesture_desc = gesture_descriptions[class_name]
    
    prompt = (
        f"Generate a new image of a human hand showing a {gesture_desc} (sample variation {sample_index}). "
        f"{variation}. "
        f"The hand should have diverse skin color and tone - use different skin tones, hand sizes, and appearances and have different pov, angles, and positions."
        f"BACKGROUND REQUIREMENTS: Use diverse and varied backgrounds - include different environments such as: "
        f"indoor settings (offices, homes, studios with plain or textured walls), outdoor settings (parks, streets, nature), people in the background, "
        f"neutral backgrounds (solid colors, gradients, abstract patterns), and various lighting conditions. "
        f"Ensure the background is distinct from the hand to maintain clear visual separation. "
        f"CRITICAL: The generated image MUST have a human hand. In different images, the hand should be in different positions and orientations but with the same gesture. "
        f"IMPORTANT: Do NOT draw any bounding boxes on the image itself. The image should be clean without any annotations or bounding boxes drawn on it. "
        f"BOUNDING BOX COORDINATES REQUIREMENTS (JSON format only): "
        f"1. Provide the bounding box coordinates in JSON format only - do NOT draw anything on the image. "
        f"2. The bounding box coordinates must define a PERFECT RECTANGLE - a tight rectangular box around the hand. "
        f"3. The bounding box must be EXTREMELY TIGHT around the hand - fit as snugly as possible with ABSOLUTE MINIMAL padding (just 1-2 pixels maximum). "
        f"4. The box coordinates must tightly wrap around the fingertips, thumb, palm edges, and wrist - no extra space. "
        f"5. The width and height should be as small as possible while still fully containing the entire hand. "
        f"6. The rectangular box must have 90-degree corners - perfectly square corners, not rounded or angled. "
        f"7. CRITICAL: The bounding box must contain ONLY ONE HAND/OBJECT - do NOT combine multiple hands, objects, or body parts in a single bounding box. "
        f"   Each bounding box must tightly enclose exactly one hand showing the gesture, with no other objects or body parts included. "
        f"   For this dataset we need only ONE hand per image. "
        f"8. The JSON coordinates, when used to draw a bounding box, should create a tight rectangular box that perfectly fits around the hand. "
        f"CRITICAL: You must return the generated image (without any bounding boxes drawn on it) "
        f"and provide the tight bounding box coordinates in JSON format. "
        f"The JSON must contain a 'labels' array with at least one object having: "
        f"'class_name' (string, set to '{class_name}'), "
        f"'format' (string, set to 'yolo'), "
        f"'center_x', 'center_y', 'width', 'height' (all numbers in YOLO normalized coordinates, 0-1 range). "
        f"The JSON coordinates must define a tight rectangular bounding box that can be drawn on the image when needed. "
        f"IMPORTANT: The image should contain only ONE hand showing the gesture - do not include multiple hands or combine multiple objects in the image."
    )
    return prompt


def extract_image_and_labels(response_data: dict):
    """Extract image and labels from API response."""
    base64_image_string = None
    labels_json = None
    
    try:
        parts = response_data['candidates'][0]['content']['parts']
        
        for part in parts:
            # Check for image data
            if 'inlineData' in part:
                base64_image_string = part['inlineData']['data']
            elif 'inline_data' in part:
                base64_image_string = part['inline_data']['data']
            
            # Check for text/JSON labels
            if 'text' in part:
                text = part['text']
                try:
                    labels_json = json.loads(text)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                    if json_match:
                        try:
                            labels_json = json.loads(json_match.group(1))
                        except json.JSONDecodeError:
                            pass
                    if not labels_json:
                        # Try to find JSON object directly in text
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
                        if json_match:
                            try:
                                labels_json = json.loads(json_match.group(0))
                            except json.JSONDecodeError:
                                pass
    except (KeyError, IndexError):
        return None, None
    
    return base64_image_string, labels_json


def generate_image(api_key: str, prompt: str, timeout: float = 30.0, max_retries: int = 3):
    """Generate a single image using Vertex AI API."""
    location = os.getenv("LOCATION", "us-central1")
    project_id = os.getenv("PROJECT_ID")
    
    if not project_id:
        raise ValueError("PROJECT_ID environment variable is required")
    
    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/gemini-2.5-flash-image:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.8,
            "top_k": 32,
            "top_p": 0.95,
            "response_modalities": ["TEXT", "IMAGE"],
        },
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            
            if not response.ok:
                status_code = response.status_code
                print(f"    ⚠️ API error (attempt {attempt}/{max_retries}): {status_code}")
                
                # Handle rate limiting (429) with longer backoff
                if status_code == 429:
                    wait_time = min(60 * attempt, 300)  # Cap at 5 minutes
                    print(f"    ⏳ Rate limited. Waiting {wait_time} seconds before retry...")
                    if attempt < max_retries:
                        time.sleep(wait_time)
                        continue
                
                if attempt == max_retries:
                    response.raise_for_status()
                
                # Exponential backoff for other errors
                time.sleep(2 ** (attempt - 1))
                continue
            
            response_data = response.json()
            image_b64, labels_json = extract_image_and_labels(response_data)
            
            if image_b64:
                return image_b64, labels_json
            else:
                print(f"    ⚠️ No image in response (attempt {attempt}/{max_retries})")
                if attempt == max_retries:
                    return None, None
                time.sleep(2 ** (attempt - 1))
                
        except Exception as e:
            print(f"    ⚠️ Error (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                return None, None
            # Longer wait for rate limiting errors
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = min(60 * attempt, 300)
                print(f"    ⏳ Rate limited. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                time.sleep(2 ** (attempt - 1))
    
    return None, None


def save_yolo_label(label_path: Path, class_id: int, center_x: float, center_y: float, width: float, height: float):
    """Save YOLO format label file."""
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def get_next_index_for_class(class_name: str, images_dir: Path) -> int:
    """Find the next available index for a class by checking existing files."""
    if not images_dir.exists():
        return 1
    
    # Pattern: class_name_XXX.jpg or .jpeg
    existing_files = list(images_dir.glob(f"{class_name}_*.jpg")) + list(images_dir.glob(f"{class_name}_*.jpeg"))
    
    if not existing_files:
        return 1
    
    # Extract indices from filenames
    indices = []
    for file in existing_files:
        # Extract number from filename like "go_up_001.jpg" or "go_up_001.jpeg"
        match = re.search(rf"{re.escape(class_name)}_(\d+)\.(jpg|jpeg)", file.name)
        if match:
            indices.append(int(match.group(1)))
    
    if not indices:
        return 1
    
    # Return next index after the highest existing one
    return max(indices) + 1


def main():
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment")
        sys.exit(1)
    
    dataset_config = {class_name: 10 for class_name in CLASS_NAMES}
    
    # Create output directories
    output_dir = Path("test_data")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting dataset generation")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    total_generated = 0
    total_failed = 0
    
    # Generate images for each class
    for class_name, count in dataset_config.items():
        class_id = CLASS_TO_ID[class_name]
        print(f"\n📦 Generating {class_name} (class_id={class_id}): {count} samples")
        
        # Find next available index for this class (continues from existing files)
        global_index = get_next_index_for_class(class_name, images_dir)
        if global_index > 1:
            print(f"   📌 Continuing from index {global_index:03d} (found existing files)")
        
        for i in range(count):
            sample_index = global_index
            prompt = build_prompt(class_name, sample_index)
            
            print(f"      [{sample_index:03d}] Generating...", end=" ", flush=True)
            
            image_b64, labels_json = generate_image(api_key, prompt)
            
            if not image_b64:
                print("❌ Failed (no image)")
                total_failed += 1
                continue
            
            # Extract labels
            labels = []
            if labels_json:
                if "labels" in labels_json:
                    labels = labels_json["labels"]
                elif isinstance(labels_json, list):
                    labels = labels_json
                else:
                    labels = [labels_json]
            
            if not labels:
                print("❌ Failed (no labels)")
                total_failed += 1
                continue
            
            # Get the first label (should be the hand)
            label = labels[0]
            center_x = float(label.get("center_x", 0.5))
            center_y = float(label.get("center_y", 0.5))
            width = float(label.get("width", 0.3))
            height = float(label.get("height", 0.3))
            
            # Validate coordinates
            if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                print("❌ Failed (invalid coordinates)")
                total_failed += 1
                continue
            
            # Save image
            try:
                image_data = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_data))
                image_filename = f"{class_name}_{sample_index:03d}.jpg"
                image_path = images_dir / image_filename
                image.save(image_path, "JPEG")
                
                # Save label
                label_filename = f"{class_name}_{sample_index:03d}.txt"
                label_path = labels_dir / label_filename
                save_yolo_label(label_path, class_id, center_x, center_y, width, height)
                
                print(f"✅ Saved ({image.size[0]}x{image.size[1]}, bbox: {width:.3f}x{height:.3f})")
                total_generated += 1
                global_index += 1
                
            except Exception as e:
                print(f"❌ Failed (save error: {e})")
                total_failed += 1
            
            # Delay to avoid rate limiting (increased for Vertex AI)
            time.sleep(2.0)
    
    print("\n" + "="*60)
    print("✅ Dataset generation complete!")
    print(f"   Generated: {total_generated} images")
    print(f"   Failed: {total_failed} attempts")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
