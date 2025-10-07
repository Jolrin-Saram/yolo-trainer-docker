# This file will contain the training logic.
# It will be called by run_training_ui.py

import sys
import yaml
from ultralytics import YOLO
import torch

# Map simple names to strings that instantiate the class.
ACTIVATION_MAP = {
    'relu': 'torch.nn.ReLU()',
    'leakyrelu': 'torch.nn.LeakyReLU()',
    'mish': 'torch.nn.Mish()',
    'silu': 'torch.nn.SiLU()',
}

def start_training(dataset_yaml, model_weights, model_yaml, epochs, activation, device, save_dir, lr0, dropout, batch_size):
    """
    Starts the YOLOv8 training process.
    """
    try:
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

        # Load the base model yaml
        with open(model_yaml, 'r', encoding='utf-8') as f:
            model_config = yaml.safe_load(f)

        # Get the fully qualified name for the activation function
        activation_fqn = ACTIVATION_MAP.get(activation.lower())
        if not activation_fqn:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Set the activation to the fully qualified name string
        model_config['activation'] = activation_fqn

        # Write to a temporary yaml file
        temp_config_path = 'temp_config.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, sort_keys=False)

        print(f"Generated temporary config with activation '{activation}' at {temp_config_path}")

        # Initialize model from the modified yaml
        model = YOLO(temp_config_path)

        # Train the model, using the 'model' argument for pretrained weights
        results = model.train(
            data=dataset_yaml,
            model=model_weights, # Pass pretrained weights via the 'model' argument
            epochs=int(epochs),
            device=device,
            project=save_dir,
            name='train',
            lr0=float(lr0),
            dropout=float(dropout),
            batch=int(batch_size)
        )

        print("Training finished successfully.")
        print(f"Results saved to: {results.save_dir}")

    except Exception as e:
        print(f"An error occurred during training: {e}", file=sys.stderr)
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