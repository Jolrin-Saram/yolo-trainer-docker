# This file will contain the auto-labeling logic.
# It will be called by run_training_ui.py

import sys
import os
from ultralytics import YOLO
from tqdm import tqdm

def auto_label(images_path, model_path, conf_threshold, iou_threshold, save_path):
    """
    Performs auto-labeling on a directory of images and saves annotations to a specified path.

    Args:
        images_path (str): Path to the directory containing images.
        model_path (str): Path to the trained model (.pt file).
        conf_threshold (str): Confidence threshold for predictions.
        iou_threshold (str): IoU threshold for Non-Max Suppression.
        save_path (str): Directory to save the annotation .txt files.
    """
    try:
        print("Starting auto-labeling with:")
        print(f"  - Images Path: {images_path}")
        print(f"  - Model: {model_path}")
        print(f"  - Confidence: {conf_threshold}")
        print(f"  - IoU: {iou_threshold}")
        print(f"  - Save Path: {save_path}")

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        model = YOLO(model_path)

        image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        total_files = len(image_files)
        for i, image_file in enumerate(tqdm(image_files, desc="Auto-labeling progress")):
            # Run prediction
            results = model.predict(source=image_file, conf=float(conf_threshold), iou=float(iou_threshold))

            # Prepare label file path in the specified save directory
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(save_path, f"{base_name}.txt")

            with open(label_file, 'w') as f:
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        x_center, y_center, width, height = box.xywhn[0]
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Progress reporting for the UI
            progress_percent = int(((i + 1) / total_files) * 100)
            print(f"Progress:{progress_percent}%")
            sys.stdout.flush()

        print("Auto-labeling finished successfully.")

    except Exception as e:
        print(f"An error occurred during auto-labeling: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python auto_labeler.py <images_path> <model_path> <conf_threshold> <iou_threshold> <save_path>", file=sys.stderr)
        sys.exit(1)

    images_path = sys.argv[1]
    model_path = sys.argv[2]
    conf = sys.argv[3]
    iou = sys.argv[4]
    save_path = sys.argv[5]

    auto_label(images_path, model_path, conf, iou, save_path)
