#!/usr/bin/env python3
"""
Generate synthetic dataset using Gemini image generation API.
Creates YOLO-format dataset with images and labels.

IMPORTANT: Bounding boxes must be:
- Perfect rectangles (straight horizontal/vertical lines, 90-degree corners)
- Consistent color (bright red RGB: 255, 0, 0) across all images
- Consistent thickness (3 pixels) for visual uniformity
- Tightly fitted around the hand with minimal padding
- Contain ONLY ONE hand/object per bounding box (no multiple objects combined)

BACKGROUND REQUIREMENTS:
- Use diverse backgrounds (indoor, outdoor, neutral, various lighting)
- Ensure clear visual separation between hand and background
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


def build_prompt(class_name: str, gesture_type: str, sample_index: int) -> str:
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
        "go_up": {
            "thumbs_up": "thumbs up gesture",
            "index_up": "index finger pointing up gesture",
        },
        "go_down": {
            "thumbs_down": "thumbs down gesture",
            "index_down": "index finger pointing down gesture",
        },
        "rotate": {
            "stop": "stop hand sign (palm facing forward)",
        },
    }
    
    gesture_desc = gesture_descriptions[class_name][gesture_type]
    
    prompt = (
        f"Generate a new image of a human hand showing a {gesture_desc} (sample variation {sample_index}). "
        f"{variation}. "
        f"The hand should have diverse skin color and tone - use different skin tones, hand sizes, and appearances. "
        f"BACKGROUND REQUIREMENTS: Use diverse and varied backgrounds - include different environments such as: "
        f"indoor settings (offices, homes, studios with plain or textured walls), outdoor settings (parks, streets, nature), "
        f"neutral backgrounds (solid colors, gradients, abstract patterns), and various lighting conditions. "
        f"Ensure the background is distinct from the hand to maintain clear visual separation. "
        f"CRITICAL: The generated image MUST have a tight bounding box drawn ON the image itself around the hand. "
        f"IMPORTANT BOUNDING BOX REQUIREMENTS FOR DATASET UNIFORMITY: "
        f"1. The bounding box MUST be a PERFECT RECTANGLE - use straight horizontal and vertical lines only, NO random shapes, curves, or irregular polygons. It should be in front of the background."
        f"2. Use a CONSISTENT COLOR for all bounding boxes - use bright red (RGB: 255, 0, 0) for uniformity across the entire dataset. "
        f"3. Use a CONSISTENT THICKNESS - draw the bounding box with a line thickness of exactly 3 pixels for all images to maintain visual consistency. "
        f"4. The bounding box must be EXTREMELY TIGHT around the hand - fit as snugly as possible with ABSOLUTE MINIMAL padding (just 1-2 pixels maximum). "
        f"5. The box must tightly wrap around the fingertips, thumb, palm edges, and wrist - no extra space. "
        f"6. The width and height should be as small as possible while still fully containing the entire hand. "
        f"7. The rectangular box must have 90-degree corners - perfectly square corners, not rounded or angled. "
        f"8. CRITICAL: The bounding box must contain ONLY ONE HAND/OBJECT - do NOT combine multiple hands, objects, or body parts in a single bounding box. "
        f"   Each bounding box must tightly enclose exactly one hand showing the gesture, with no other objects or body parts included. "
        f"   If there are multiple hands or objects in the image, draw separate bounding boxes for each, but for this dataset we need only ONE hand per image. "
        f"This is the most important requirement - the bounding box ON THE IMAGE must be extremely precise, tight, rectangular, uniform in appearance, and contain only one hand. "
        f"CRITICAL: You must return the generated image with the rectangular bounding box drawn on it as part of your response. "
        f"Additionally, provide the bounding box coordinates in JSON format. "
        f"The JSON must contain a 'labels' array with at least one object having: "
        f"'class_name' (string, set to '{class_name}'), "
        f"'format' (string, set to 'yolo'), "
        f"'center_x', 'center_y', 'width', 'height' (all numbers in YOLO normalized coordinates, 0-1 range). "
        f"The JSON coordinates must match exactly the rectangular bounding box drawn on the image. "
        f"IMPORTANT: The image should contain only ONE hand showing the gesture - do not include multiple hands or combine multiple objects in the bounding box."
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
    """Generate a single image using Gemini API."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.8,
            "top_k": 32,
            "top_p": 0.95,
            "response_modalities": ["TEXT", "IMAGE"],
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            
            if not response.ok:
                print(f"    ⚠️ API error (attempt {attempt}/{max_retries}): {response.status_code}")
                if attempt == max_retries:
                    response.raise_for_status()
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
            time.sleep(2 ** (attempt - 1))
    
    return None, None


def save_yolo_label(label_path: Path, class_id: int, center_x: float, center_y: float, width: float, height: float):
    """Save YOLO format label file."""
    with open(label_path, 'w') as f:
        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")


def main():
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment")
        sys.exit(1)
    
    # Class mapping
    class_to_id = {"go_up": 0, "go_down": 1, "rotate": 2}
    
    # Dataset configuration
    dataset_config = {
        "go_up": [
            ("thumbs_up", 70),
            ("index_up", 70),
        ],
        "go_down": [
            ("thumbs_down", 70),
            ("index_down", 70),
        ],
        "rotate": [
            ("stop", 70),
        ],
    }
    
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
    for class_name, variants in dataset_config.items():
        class_id = class_to_id[class_name]
        print(f"\n📦 Generating {class_name} (class_id={class_id})")
        print(f"   Variants: {variants}")
        
        global_index = 1  # Global index for this class
        
        for gesture_type, count in variants:
            print(f"\n   🎯 {gesture_type}: {count} samples")
            
            for i in range(count):
                sample_index = global_index
                prompt = build_prompt(class_name, gesture_type, sample_index)
                
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
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
    
    print("\n" + "="*60)
    print("✅ Dataset generation complete!")
    print(f"   Generated: {total_generated} images")
    print(f"   Failed: {total_failed} attempts")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {labels_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
