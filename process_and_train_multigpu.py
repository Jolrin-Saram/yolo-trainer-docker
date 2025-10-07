# This file will contain the training logic.
# Multi-GPU Enhanced Version
# It will be called by run_training_ui.py

import sys
import yaml
from ultralytics import YOLO
import torch
import os
from pathlib import Path

# Map simple names to strings that instantiate the class.
ACTIVATION_MAP = {
    'relu': 'torch.nn.ReLU()',
    'leakyrelu': 'torch.nn.LeakyReLU()',
    'mish': 'torch.nn.Mish()',
    'silu': 'torch.nn.SiLU()',
}

def parse_device_string(device_str):
    """
    Parse device string to support multi-GPU configurations.
    Examples:
        "0" -> "0" (single GPU)
        "0,1,2,3" -> "0,1,2,3" (multi-GPU with 4 GPUs)
        "auto" -> auto-detect all available GPUs
        "cpu" -> "cpu"

    Returns:
        str: Validated device string
    """
    if device_str.lower() == "auto":
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            device_str = ",".join(str(i) for i in range(gpu_count))
            print(f"[Auto-detected {gpu_count} GPU(s): {device_str}]")
        else:
            device_str = "cpu"
            print("[No GPU detected, using CPU]")
    elif device_str.lower() == "cpu":
        device_str = "cpu"
    else:
        # Validate GPU indices
        if device_str != "cpu":
            try:
                gpu_indices = [int(x.strip()) for x in device_str.split(",")]
                available_gpus = torch.cuda.device_count()
                for idx in gpu_indices:
                    if idx >= available_gpus:
                        raise ValueError(f"GPU {idx} not available. Only {available_gpus} GPU(s) detected.")
                print(f"[Using GPU(s): {device_str}]")
            except ValueError as e:
                print(f"[Error parsing device string: {e}]", file=sys.stderr)
                raise

    return device_str

def start_training(dataset_yaml, model_weights, model_yaml, epochs, activation, device, save_dir, lr0, dropout, batch_size):
    """
    Starts the YOLOv8 training process with multi-GPU support.

    Multi-GPU Training:
    - Ultralytics YOLO automatically uses DDP (DistributedDataParallel) when multiple GPUs are specified
    - Batch size is distributed across GPUs (e.g., batch_size=32 on 4 GPUs = 8 per GPU)
    - Training speed scales nearly linearly with GPU count
    """
    try:
        # Parse and validate device string
        device = parse_device_string(device)

        print(f"Starting training with:")
        print(f"  - Dataset YAML: {dataset_yaml}")
        print(f"  - Model YAML: {model_yaml}")
        print(f"  - Pretrained Weights: {model_weights}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Activation: {activation}")
        print(f"  - Device: {device}")
        print(f"  - Save Directory: {save_dir}")
        print(f"  - Learning Rate: {lr0}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Batch Size: {batch_size}")

        # Check if multi-GPU training
        if "," in str(device):
            gpu_count = len(device.split(","))
            print(f"\n{'='*60}")
            print(f"MULTI-GPU TRAINING ENABLED")
            print(f"{'='*60}")
            print(f"Number of GPUs: {gpu_count}")
            print(f"Total Batch Size: {batch_size}")
            print(f"Batch Size per GPU: ~{int(batch_size) // gpu_count}")
            print(f"Training Method: DDP (DistributedDataParallel)")
            print(f"{'='*60}\n")

        # Load the base model yaml
        with open(model_yaml, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        # Get the fully qualified name for the activation function
        activation_fqn = ACTIVATION_MAP.get(activation.lower())
        if not activation_fqn:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Set the activation to the fully qualified name string
        model_config['activation'] = activation_fqn

        # Write to a temporary yaml file (use script directory for portability)
        script_dir = Path(__file__).parent.resolve()
        temp_config_path = script_dir / 'temp_config.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, sort_keys=False)

        print(f"Generated temporary config with activation '{activation}' at {temp_config_path}")

        # Initialize model from the modified yaml
        model = YOLO(str(temp_config_path))

        # Train the model with multi-GPU support
        # Ultralytics automatically handles multi-GPU with DDP (DistributedDataParallel)
        results = model.train(
            data=dataset_yaml,
            model=model_weights,  # Pass pretrained weights via the 'model' argument
            epochs=int(epochs),
            device=device,  # Now supports multi-GPU (e.g., "0,1,2,3")
            project=save_dir,
            name='train',
            lr0=float(lr0),
            dropout=float(dropout),
            batch=int(batch_size)
        )

        print("\n" + "="*60)
        print("Training finished successfully!")
        print(f"Results saved to: {results.save_dir}")
        print("="*60)

    except Exception as e:
        print(f"\nAn error occurred during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 11:
        print("Usage: python process_and_train.py <dataset_yaml> <model_weights> <model_yaml> <epochs> <activation> <device> <save_dir> <lr0> <dropout> <batch_size>", file=sys.stderr)
        sys.exit(1)

    dataset_yaml_path = sys.argv[1]
    model_weights_path = sys.argv[2]
    model_yaml_path = sys.argv[3]
    num_epochs = sys.argv[4]
    activation_function = sys.argv[5]
    device_to_use = sys.argv[6]
    save_dir_path = sys.argv[7]
    lr0_val = sys.argv[8]
    dropout_val = sys.argv[9]
    batch_size_val = sys.argv[10]

    start_training(dataset_yaml_path, model_weights_path, model_yaml_path, num_epochs, activation_function, device_to_use, save_dir_path, lr0_val, dropout_val, batch_size_val)
