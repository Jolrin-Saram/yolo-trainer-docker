# prepare_dataset.py - Enhanced with custom output directory and skip existing files
import os
import sys
import yaml
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_dataset_split(dataset_root, class_file, val_ratio, output_dir=None):
    """
    Finds all image-label pairs, merges them, splits them into train/val sets,
    creates a new structured directory, and generates a data.yaml file.

    Args:
        dataset_root (str): Root directory containing images and labels
        class_file (str): Path to classes.txt file
        val_ratio (str/float): Validation ratio (0-100 or 0.0-1.0)
        output_dir (str, optional): Custom output directory. If None, uses 'dataset_prepared'

    Returns:
        str: Path to generated data.yaml file
    """
    dataset_root = Path(dataset_root)
    class_file = Path(class_file)
    val_ratio = float(val_ratio) / 100.0 if float(val_ratio) > 1.0 else float(val_ratio)

    # --- 1. Find all images and match with labels ---
    print("="*60)
    print("DATASET PREPARATION - Enhanced Version")
    print("="*60)
    print("\nStep 1/6: Searching for image files...")
    sys.stdout.flush()

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_root.rglob(ext))

    print(f"✓ Found {len(image_files)} total image files")
    sys.stdout.flush()

    print("\nStep 2/6: Matching images with labels...")
    sys.stdout.flush()
    pairs = []
    for img_path in tqdm(image_files, desc="Matching pairs"):
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            pairs.append((img_path, label_path))

    if not pairs:
        raise ValueError(f"No matching image-label pairs found in {dataset_root}")

    print(f"✓ Found {len(pairs)} image-label pairs")
    print(f"  - Images without labels: {len(image_files) - len(pairs)}")
    sys.stdout.flush()

    # --- 2. Shuffle and split ---
    print("\nStep 3/6: Shuffling and splitting dataset...")
    sys.stdout.flush()
    random.shuffle(pairs)
    split_index = int(len(pairs) * (1 - val_ratio))
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]
    print(f"✓ Split complete:")
    print(f"  - Training set: {len(train_pairs)} pairs ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"  - Validation set: {len(val_pairs)} pairs ({len(val_pairs)/len(pairs)*100:.1f}%)")
    sys.stdout.flush()

    # --- 3. Create new directory structure ---
    print("\nStep 4/6: Setting up output directory...")
    sys.stdout.flush()

    if output_dir is None:
        output_dir = Path(__file__).parent.resolve() / 'dataset_prepared'
    else:
        output_dir = Path(output_dir).resolve()
    
    print(f"✓ Output directory: {output_dir}")

    train_img_dir = output_dir / 'train' / 'images'
    train_lbl_dir = output_dir / 'train' / 'labels'
    val_img_dir = output_dir / 'val' / 'images'
    val_lbl_dir = output_dir / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- 4. Copy files to new structure (skip existing) ---
    print("\nStep 5/6: Copying files to train/val structure...")
    print("  (Skipping files that already exist)")
    sys.stdout.flush()

    total_files_to_process = (len(train_pairs) + len(val_pairs)) * 2
    processed_files = 0
    skipped_files = 0
    copied_files = 0
    last_progress_percent = -1

    def copy_with_skip(src, dst):
        """Copy file only if destination doesn't exist"""
        nonlocal skipped_files, copied_files
        if dst.exists():
            skipped_files += 1
            return False
        else:
            shutil.copyfile(src, dst)
            copied_files += 1
            return True

    print("\n  Processing training set...")
    sys.stdout.flush()
    for img, lbl in train_pairs:
        dest_img = train_img_dir / img.name
        dest_lbl = train_lbl_dir / lbl.name

        copy_with_skip(img, dest_img)
        copy_with_skip(lbl, dest_lbl)

        processed_files += 2
        progress_percent = int((processed_files / total_files_to_process) * 100)
        if progress_percent > last_progress_percent:
            print(f"PrepProgress:{progress_percent}%")
            sys.stdout.flush()
            last_progress_percent = progress_percent

    print("\n  Processing validation set...")
    sys.stdout.flush()
    for img, lbl in val_pairs:
        dest_img = val_img_dir / img.name
        dest_lbl = val_lbl_dir / lbl.name

        copy_with_skip(img, dest_img)
        copy_with_skip(lbl, dest_lbl)

        processed_files += 2
        progress_percent = int((processed_files / total_files_to_process) * 100)
        if progress_percent > last_progress_percent:
            print(f"PrepProgress:{progress_percent}%")
            sys.stdout.flush()
            last_progress_percent = progress_percent

    print(f"\n✓ File processing complete:")
    print(f"  - Copied: {copied_files} files")
    print(f"  - Skipped (already exist): {skipped_files} files")
    sys.stdout.flush()

    # --- 5. Generate data.yaml ---
    print("\nStep 6/6: Generating data.yaml file...")
    sys.stdout.flush()

    if not class_file.exists():
        raise FileNotFoundError(f"Classes file not found: {class_file}")

    with open(class_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]

    data = {
        'train': str(train_img_dir.resolve()),
        'val': str(val_img_dir.resolve()),
        'nc': len(class_names),
        'names': class_names
    }

    generated_yaml_path = output_dir / 'generated_data.yaml'
    with open(generated_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"\n{'='*60}")
    print("✅ DATASET PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Summary:")
    print(f"  - Total pairs: {len(pairs)}")
    print(f"  - Training: {len(train_pairs)} pairs")
    print(f"  - Validation: {len(val_pairs)} pairs")
    print(f"  - Classes: {len(class_names)}")
    print(f"  - Files copied: {copied_files}")
    print(f"  - Files skipped: {skipped_files}")
    print(f"\n📁 Output:")
    print(f"  - Directory: {output_dir}")
    print(f"  - YAML: {generated_yaml_path}")
    print(f"\n{'='*60}\n")

    return str(generated_yaml_path.resolve())

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Usage: python prepare_dataset.py <dataset_root> <class_file> <val_ratio> [output_dir]", file=sys.stderr)
        print("\nArguments:", file=sys.stderr)
        print("  dataset_root: Root directory containing images and labels", file=sys.stderr)
        print("  class_file: Path to classes.txt file", file=sys.stderr)
        print("  val_ratio: Validation ratio (0-100)", file=sys.stderr)
        print("  output_dir: (Optional) Custom output directory", file=sys.stderr)
        sys.exit(1)

    try:
        dataset_root = sys.argv[1]
        class_file = sys.argv[2]
        val_ratio = sys.argv[3]
        output_dir = sys.argv[4] if len(sys.argv) == 5 else None

        yaml_path = prepare_dataset_split(dataset_root, class_file, val_ratio, output_dir)
        print(yaml_path)  # Print the path for the calling script to capture
    except Exception as e:
        print(f"Error preparing dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)