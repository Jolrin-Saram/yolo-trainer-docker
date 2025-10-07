# This file will contain the auto-labeling logic.
# Multi-GPU Enhanced Version (uses single GPU for inference)
# It will be called by run_training_ui.py

import sys
import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def parse_device_string(device_str):
    """
    Parse device string to support GPU configurations.
    For auto-labeling, we use the first GPU in the list for inference.

    Examples:
        "0" -> "0" (single GPU)
        "0,1,2,3" -> "0" (use first GPU)
        "auto" -> "0" or "cpu"
        "cpu" -> "cpu"
    """
    if device_str.lower() == "auto":
        if torch.cuda.is_available():
            device_str = "0"
            print(f"[Auto-detected GPU, using GPU 0 for inference]")
        else:
            device_str = "cpu"
            print("[No GPU detected, using CPU]")
    elif device_str.lower() == "cpu":
        device_str = "cpu"
    else:
        # If multi-GPU string provided (e.g., "0,1,2,3"), use first GPU for inference
        if "," in device_str:
            original = device_str
            device_str = device_str.split(",")[0]
            print(f"[Multi-GPU string '{original}' detected, using GPU {device_str} for auto-labeling]")
        # Validate GPU index
        if device_str != "cpu":
            try:
                gpu_idx = int(device_str)
                available_gpus = torch.cuda.device_count()
                if gpu_idx >= available_gpus:
                    raise ValueError(f"GPU {gpu_idx} not available. Only {available_gpus} GPU(s) detected.")
                print(f"[Using GPU {device_str} for inference]")
            except ValueError as e:
                print(f"[Error parsing device string: {e}]", file=sys.stderr)
                raise

    return device_str

def auto_label(images_path, model_path, conf_threshold, iou_threshold, save_path, device="auto"):
    """
    Performs auto-labeling on a directory of images and saves annotations to a specified path.

    Args:
        images_path (str): Path to the directory containing images.
        model_path (str): Path to the trained model (.pt file).
        conf_threshold (str): Confidence threshold for predictions.
        iou_threshold (str): IoU threshold for Non-Max Suppression.
        save_path (str): Directory to save the annotation .txt files.
        device (str): Device to use for inference (default: "auto").
    """
    try:
        # Parse device string
        device = parse_device_string(device)

        print("="*60)
        print("Starting auto-labeling with:")
        print(f"  - Images Path: {images_path}")
        print(f"  - Model: {model_path}")
        print(f"  - Confidence: {conf_threshold}")
        print(f"  - IoU: {iou_threshold}")
        print(f"  - Save Path: {save_path}")
        print(f"  - Device: {device}")
        print("="*60)

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Load model with device specification
        model = YOLO(model_path)

        # Find all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        image_files = [
            os.path.join(images_path, f)
            for f in os.listdir(images_path)
            if f.lower().endswith(image_extensions)
        ]

        if not image_files:
            print("No image files found in the specified directory.", file=sys.stderr)
            sys.exit(1)

        total_files = len(image_files)
        print(f"\nFound {total_files} image(s) to process\n")

        labeled_count = 0
        empty_count = 0

        for i, image_file in enumerate(tqdm(image_files, desc="Auto-labeling progress")):
            # Run prediction with device specification
            results = model.predict(
                source=image_file,
                conf=float(conf_threshold),
                iou=float(iou_threshold),
                device=device,
                verbose=False  # Reduce console output
            )

            # Prepare label file path in the specified save directory
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(save_path, f"{base_name}.txt")

            # Write labels
            box_count = 0
            with open(label_file, 'w') as f:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            x_center, y_center, width, height = box.xywhn[0]
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            box_count += 1

            if box_count > 0:
                labeled_count += 1
            else:
                empty_count += 1

            # Progress reporting for the UI
            progress_percent = int(((i + 1) / total_files) * 100)
            print(f"Progress:{progress_percent}%")
            sys.stdout.flush()

        print("\n" + "="*60)
        print("Auto-labeling finished successfully!")
        print(f"Total images processed: {total_files}")
        print(f"Images with labels: {labeled_count}")
        print(f"Images without labels: {empty_count}")
        print("="*60)

    except Exception as e:
        print(f"\nAn error occurred during auto-labeling: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print("Usage: python auto_labeler.py <images_path> <model_path> <conf_threshold> <iou_threshold> <save_path> [device]", file=sys.stderr)
        sys.exit(1)

    images_path = sys.argv[1]
    model_path = sys.argv[2]
    conf = sys.argv[3]
    iou = sys.argv[4]
    save_path = sys.argv[5]
    device = sys.argv[6] if len(sys.argv) == 7 else "auto"

    auto_label(images_path, model_path, conf, iou, save_path, device)
