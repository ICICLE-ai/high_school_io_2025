#!/usr/bin/env python3
"""
YOLO Ultralytics Training Script
Automatically splits data into train/val/test and trains a YOLO model
"""

import shutil
import random
from collections import defaultdict
from pathlib import Path

from gesture_config import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def extract_class_name(image_path: Path) -> str:
    """Extract the snake_case class name from filenames like class_name_001.jpg."""
    stem_parts = image_path.stem.rsplit("_", 1)
    if len(stem_parts) == 2 and stem_parts[1].isdigit():
        return stem_parts[0]
    return image_path.stem


def build_stratified_splits(image_files, train_ratio, val_ratio, test_ratio, seed=42):
    """Split each class independently so every split stays balanced."""
    grouped_images = defaultdict(list)
    for image_path in image_files:
        grouped_images[extract_class_name(image_path)].append(image_path)

    rng = random.Random(seed)
    splits = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_summary = {}

    for class_name, class_images in sorted(grouped_images.items()):
        class_images = sorted(class_images)
        rng.shuffle(class_images)

        n_total = len(class_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_images = class_images[:n_train]
        val_images = class_images[n_train:n_train + n_val]
        test_images = class_images[n_train + n_val:]

        splits["train"].extend(train_images)
        splits["val"].extend(val_images)
        splits["test"].extend(test_images)
        split_summary[class_name] = {
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
            "total": n_total,
        }

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name], key=lambda path: path.name)

    return splits, split_summary


def setup_dataset(data_dir="test_data", output_dir="yolo_dataset", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Organize YOLO dataset with train/val/test splits
    
    Args:
        data_dir: Directory containing images/ and labels/ folders
        output_dir: Output directory for YOLO dataset structure
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    image_files = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"Found {len(image_files)} images")

    splits, split_summary = build_stratified_splits(
        image_files=image_files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    print(
        f"Split: {len(splits['train'])} train, "
        f"{len(splits['val'])} val, {len(splits['test'])} test"
    )
    print("Per-class distribution:")
    for class_name, stats in split_summary.items():
        print(
            f"  - {class_name}: {stats['train']} train, {stats['val']} val, "
            f"{stats['test']} test (total={stats['total']})"
        )

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    for split_name, split_images in splits.items():
        split_images_dir = output_path / split_name / "images"
        split_labels_dir = output_path / split_name / "labels"
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in split_images:
            # Copy image
            shutil.copy2(img_path, split_images_dir / img_path.name)
            
            # Copy corresponding label
            label_name = img_path.stem + ".txt"
            label_path = labels_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, split_labels_dir / label_name)
            else:
                print(f"Warning: Label not found for {img_path.name}")
    
    print(f"Dataset structure created in {output_dir}/")
    return output_path


def train_yolo(
    data_yaml_path,
    model_size="n",  # n, s, m, l, x
    epochs=100,
    patience=None,
    imgsz=640,
    batch=16,
    device=None,  # Auto-detect if None
    project="runs/detect",
    name="yolo_training"
):
    """
    Train YOLO model using Ultralytics
    
    Args:
        data_yaml_path: Path to dataset.yaml file
        model_size: YOLO model size (nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        patience: Early stopping patience. Defaults to `epochs` to avoid stopping short.
        imgsz: Image size for training
        batch: Batch size
        device: Device to use (mps for Mac M1/M2, cpu, or cuda)
        project: Project directory
        name: Experiment name
    """
    # Initialize model
    from ultralytics import YOLO

    model = YOLO(f"yolov8{model_size}.pt")  # Load pretrained weights
    
    # Auto-detect device if not specified
    if device is None:
        try:
            import torch
            if torch.backends.mps.is_available():
                device = "mps"
                print("Using MPS (Metal Performance Shaders) for Mac GPU acceleration")
            else:
                device = "cpu"
                print("Using CPU (MPS not available)")
        except Exception:
            device = "cpu"
            print("Using CPU")

    if patience is None:
        patience = epochs
    
    # Train the model
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save=True,
        plots=True,
        val=True,
    )
    
    print("\nTraining completed!")
    print(f"Results saved in {project}/{name}/")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.val(data=str(data_yaml_path), split="test")
    print(f"Test mAP50: {test_results.box.map50:.4f}")
    print(f"Test mAP50-95: {test_results.box.map:.4f}")
    
    return model, results


def create_data_yaml(dataset_dir, output_path="dataset.yaml", class_names=None):
    """
    Create YOLO dataset.yaml file
    
    Args:
        dataset_dir: Path to dataset directory (with train/val/test subdirs)
        output_path: Path to save dataset.yaml
        class_names: List of class names (default: go_up=0, go_down=1, rotate=2)
                     Order determines class IDs: index 0 = class 0, index 1 = class 1, etc.
    """
    if class_names is None:
        class_names = CLASS_NAMES  # Order determines class IDs
    
    dataset_path = Path(dataset_dir).absolute()
    
    yaml_content = f"""# YOLO Dataset Configuration
path: {dataset_path}
train: train/images
val: val/images
test: test/images

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(output_path, "w") as f:
        f.write(yaml_content)
    
    print(f"Created dataset.yaml at {output_path}")
    return Path(output_path)


def main():
    """Main training pipeline"""
    # Configuration
    DATA_DIR = "test_data"
    OUTPUT_DATASET_DIR = "yolo_dataset"
    DATASET_YAML = "dataset.yaml"
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Training parameters
    MODEL_SIZE = "n"  # Use 'n' for nano (fastest), 's' for small, 'm' for medium, etc.
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 16  # Adjust based on your Mac's memory
    
    # Class names (order determines class IDs)
    print("=" * 60)
    print("YOLO Ultralytics Training Script")
    print("=" * 60)
    
    # Step 1: Setup dataset with train/val/test splits
    print("\n[1/3] Setting up dataset structure...")
    dataset_path = setup_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DATASET_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    print(f"Dataset prepared at {dataset_path}")
    
    # Step 2: Create dataset.yaml
    print("\n[2/3] Creating dataset.yaml...")
    yaml_path = create_data_yaml(
        dataset_dir=OUTPUT_DATASET_DIR,
        output_path=DATASET_YAML,
        class_names=CLASS_NAMES
    )
    
    # Step 3: Train model
    print("\n[3/3] Starting training...")
    
    # Device will be auto-detected in train_yolo function
    model, results = train_yolo(
        data_yaml_path=yaml_path,
        model_size=MODEL_SIZE,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=None,  # Auto-detect (MPS for Mac, CPU otherwise)
        project="runs/detect",
        name="yolo_training"
    )
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("Best model: runs/detect/yolo_training/weights/best.pt")
    print("Last model: runs/detect/yolo_training/weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
