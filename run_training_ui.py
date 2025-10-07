
import sys
import json
import os
import torch
import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QTabWidget, QFileDialog, QProgressBar, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess

# Placeholder for process_and_train.py and auto_labeler.py
# These would contain the actual logic for training and auto-labeling.
# For now, we'll just simulate their execution.

class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    prep_progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, dataset_folder, classes_file, val_ratio, save_dir, model_size, activation, epochs, device, lr0, dropout, batch_size, prep_output_dir=None):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.classes_file = classes_file
        self.val_ratio = val_ratio
        self.save_dir = save_dir
        self.model_size = model_size
        self.activation = activation
        self.epochs = epochs
        self.device = device
        self.lr0 = lr0
        self.dropout = dropout
        self.batch_size = batch_size
        self.prep_output_dir = prep_output_dir

    def run(self):
        try:
            # Step 1: Prepare dataset and get the path to the generated data.yaml
            self.log_signal.emit(f"Analyzing dataset folder: {self.dataset_folder}")

            # Build command with optional output_dir
            prep_cmd = [sys.executable, "prepare_dataset.py", self.dataset_folder, self.classes_file, self.val_ratio]
            if self.prep_output_dir:
                prep_cmd.append(self.prep_output_dir)
                self.log_signal.emit(f"Output directory: {self.prep_output_dir}")

            prep_process = subprocess.Popen(prep_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            
            generated_yaml_path = ""
            for line in iter(prep_process.stdout.readline, ''):
                line = line.strip()
                if line.startswith("PrepProgress:"):
                    try:
                        progress = int(line.split(':')[1].replace('%', ''))
                        self.prep_progress_signal.emit(progress)
                    except:
                        pass # Ignore parsing errors
                elif line.endswith('.yaml'):
                    generated_yaml_path = line # Capture the yaml path
                else:
                    self.log_signal.emit(line)

            prep_process.stdout.close()
            stderr_output = prep_process.stderr.read()
            prep_process.wait()

            if prep_process.returncode != 0:
                self.log_signal.emit("Error preparing dataset:")
                self.log_signal.emit(stderr_output)
                self.finished_signal.emit()
                return

            self.log_signal.emit(f"Successfully generated dataset config: {generated_yaml_path}")
            self.prep_progress_signal.emit(100) # Ensure it finishes at 100%

            # Step 2: Start training
            self.log_signal.emit(f"Starting training with model size: {self.model_size}, epochs: {self.epochs}, activation: {self.activation}, device: {self.device}")
            
            model_weights = os.path.join('training_options', f"yolov8{self.model_size}.pt")
            model_yaml_name = f"yolov8{self.model_size}-{self.activation.lower()}.yaml"
            model_yaml = os.path.join('training_options', model_yaml_name)

            # If a specific activation model doesn't exist, fall back to the base model .yaml
            if not os.path.exists(model_yaml):
                model_yaml = os.path.join('training_options', f"yolov8{self.model_size}.yaml")

            train_process = subprocess.Popen([
                sys.executable, "process_and_train.py", 
                generated_yaml_path, model_weights, model_yaml, self.epochs, 
                self.activation, self.device, self.save_dir, self.lr0, self.dropout, self.batch_size
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            
            for line in iter(train_process.stdout.readline, ''):
                self.log_signal.emit(line.strip())
            
            train_process.stdout.close()
            train_process.wait()
            self.log_signal.emit("Training finished.")

        except Exception as e:
            self.log_signal.emit(f"An unexpected error occurred: {e}")
        finally:
            self.finished_signal.emit()

class AutoLabelThread(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str) # Changed from finished_signal
    finished_signal = pyqtSignal()

    def __init__(self, images_path, model_path, conf, iou, save_path):
        super().__init__()
        self.images_path = images_path
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.save_path = save_path

    def run(self):
        process = subprocess.Popen([sys.executable, "auto_labeler.py", self.images_path, self.model_path, self.conf, self.iou, self.save_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line.startswith("Progress:"):
                try:
                    progress = int(line.split(":")[1].strip().replace("%", ""))
                    self.progress_signal.emit(progress)
                except:
                    pass
            else:
                self.log_signal.emit(line)
        process.stdout.close()
        process.wait()
        self.log_signal.emit(f"Auto-labeling complete for: {self.images_path}")
        self.finished_signal.emit()


class YOLOv8TrainerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.config_file = 'config.json'
        self.log_file = None
        self.initUI()
        self.load_settings()

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def initUI(self):
        self.setWindowTitle('YOLOv8 Trainer and Auto-Labeler')
        self.setGeometry(100, 100, 800, 600)

        main_layout = QVBoxLayout()
        tabs = QTabWidget()

        # Training Tab
        train_tab = QWidget()
        train_layout = QVBoxLayout()

        # CUDA Status Label
        cuda_status_label = QLabel()
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                cuda_status_label.setText(f'CUDA Available: Yes ({gpu_count} GPU(s))')
                cuda_status_label.setStyleSheet("color: green")
            else:
                cuda_status_label.setText('CUDA Available: No')
                cuda_status_label.setStyleSheet("color: red")
        except Exception as e:
            cuda_status_label.setText('Could not check CUDA status.')
            cuda_status_label.setStyleSheet("color: orange")
        train_layout.addWidget(cuda_status_label)

        # Dataset Folder selection
        dataset_layout = QHBoxLayout()
        self.dataset_label = QLabel('Dataset Folder:')
        self.dataset_path = QLineEdit()
        self.dataset_button = QPushButton('Browse...')
        self.dataset_button.clicked.connect(self.browse_dataset_folder)
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.dataset_path)
        dataset_layout.addWidget(self.dataset_button)
        train_layout.addLayout(dataset_layout)

        # Classes file selection
        classes_file_layout = QHBoxLayout()
        self.classes_file_label = QLabel('Classes File:')
        self.classes_file_path = QLineEdit()
        self.classes_file_button = QPushButton('Browse...')
        self.classes_file_button.clicked.connect(self.browse_classes_file)
        classes_file_layout.addWidget(self.classes_file_label)
        classes_file_layout.addWidget(self.classes_file_path)
        classes_file_layout.addWidget(self.classes_file_button)
        train_layout.addLayout(classes_file_layout)

        # Validation ratio
        val_ratio_layout = QHBoxLayout()
        self.val_ratio_label = QLabel('Validation Ratio (%):')
        self.val_ratio_input = QLineEdit("20")
        val_ratio_layout.addWidget(self.val_ratio_label)
        val_ratio_layout.addWidget(self.val_ratio_input)
        train_layout.addLayout(val_ratio_layout)

        # Prepared Dataset Output Directory (NEW)
        prep_output_layout = QHBoxLayout()
        self.prep_output_label = QLabel('Prepared Dataset Dir:')
        self.prep_output_path = QLineEdit()
        self.prep_output_path.setPlaceholderText("(Optional) Leave empty for default 'dataset_prepared'")
        self.prep_output_button = QPushButton('Browse...')
        self.prep_output_button.clicked.connect(self.browse_prep_output_dir)
        prep_output_layout.addWidget(self.prep_output_label)
        prep_output_layout.addWidget(self.prep_output_path)
        prep_output_layout.addWidget(self.prep_output_button)
        train_layout.addLayout(prep_output_layout)

        # Save Directory selection
        save_dir_layout = QHBoxLayout()
        self.save_dir_label = QLabel('Training Save Directory:')
        self.save_dir_path = QLineEdit("trained_models") # Default value
        self.save_dir_button = QPushButton('Browse...')
        self.save_dir_button.clicked.connect(self.browse_save_dir)
        save_dir_layout.addWidget(self.save_dir_label)
        save_dir_layout.addWidget(self.save_dir_path)
        save_dir_layout.addWidget(self.save_dir_button)
        train_layout.addLayout(save_dir_layout)

        # Model Size
        model_size_layout = QHBoxLayout()
        self.model_size_label = QLabel('Model Size:')
        self.model_size_combo = QComboBox()
        self.populate_model_sizes()
        model_size_layout.addWidget(self.model_size_label)
        model_size_layout.addWidget(self.model_size_combo)
        train_layout.addLayout(model_size_layout)

        # Activation Function
        activation_layout = QHBoxLayout()
        self.activation_label = QLabel('Activation Function:')
        self.activation_combo = QComboBox()
        self.populate_activations()
        activation_layout.addWidget(self.activation_label)
        activation_layout.addWidget(self.activation_combo)
        train_layout.addLayout(activation_layout)

        # Learning Rate
        lr_layout = QHBoxLayout()
        self.lr_label = QLabel('Learning Rate (lr0):')
        self.lr_input = QLineEdit("0.01")
        lr_layout.addWidget(self.lr_label)
        lr_layout.addWidget(self.lr_input)
        train_layout.addLayout(lr_layout)

        # Dropout
        dropout_layout = QHBoxLayout()
        self.dropout_label = QLabel('Dropout:')
        self.dropout_input = QLineEdit("0.0")
        dropout_layout.addWidget(self.dropout_label)
        dropout_layout.addWidget(self.dropout_input)
        train_layout.addLayout(dropout_layout)

        # Batch Size
        batch_size_layout = QHBoxLayout()
        self.batch_size_label = QLabel('Batch Size:')
        self.batch_size_input = QLineEdit("16") # Default batch size
        batch_size_layout.addWidget(self.batch_size_label)
        batch_size_layout.addWidget(self.batch_size_input)
        train_layout.addLayout(batch_size_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        self.epochs_label = QLabel('Epochs:')
        self.epochs_input = QLineEdit("30")
        epochs_layout.addWidget(self.epochs_label)
        epochs_layout.addWidget(self.epochs_input)
        train_layout.addLayout(epochs_layout)

        # Device (GPU/CPU) - Enhanced Multi-GPU Support
        device_layout = QHBoxLayout()
        self.device_label = QLabel('Device:')
        self.device_combo = QComboBox()
        self.populate_device_options()
        self.device_custom_input = QLineEdit()
        self.device_custom_input.setPlaceholderText("Custom (e.g., 0,1,2,3)")
        self.device_custom_input.setVisible(False)

        # Connect combo box to show/hide custom input
        self.device_combo.currentTextChanged.connect(self.on_device_changed)

        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_combo, 2)
        device_layout.addWidget(self.device_custom_input, 1)
        train_layout.addLayout(device_layout)

        # Start Training Button
        self.start_training_button = QPushButton('Start Training')
        self.start_training_button.clicked.connect(self.start_training)
        train_layout.addWidget(self.start_training_button)

        # Dataset Prep Progress Bar
        self.prep_progress_bar = QProgressBar()
        self.prep_progress_bar.setFormat('Dataset Preparation... %p%')
        train_layout.addWidget(self.prep_progress_bar)

        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        train_layout.addWidget(self.log_area)

        train_tab.setLayout(train_layout)

        # Auto-Labeling Tab
        autolabel_tab = QWidget()
        autolabel_layout = QVBoxLayout()

        # Image/Directory selection
        images_layout = QHBoxLayout()
        self.images_label = QLabel('Images Path:')
        self.images_path = QLineEdit()
        self.images_button = QPushButton('Browse...')
        self.images_button.clicked.connect(self.browse_images)
        images_layout.addWidget(self.images_label)
        images_layout.addWidget(self.images_path)
        images_layout.addWidget(self.images_button)
        autolabel_layout.addLayout(images_layout)

        # Model selection
        model_layout_label = QHBoxLayout()
        self_model_label = QLabel('Model Path:')
        self.model_path_label = QLineEdit()
        self.model_button_label = QPushButton('Browse...')
        self.model_button_label.clicked.connect(self.browse_model_label)
        model_layout_label.addWidget(self_model_label)
        model_layout_label.addWidget(self.model_path_label)
        model_layout_label.addWidget(self.model_button_label)
        autolabel_layout.addLayout(model_layout_label)

        # Annotation save path
        save_path_layout = QHBoxLayout()
        self.save_path_label = QLabel('Save Annotations to:')
        self.save_path_input = QLineEdit()
        self.save_path_button = QPushButton('Browse...')
        self.save_path_button.clicked.connect(self.browse_save_path)
        save_path_layout.addWidget(self.save_path_label)
        save_path_layout.addWidget(self.save_path_input)
        save_path_layout.addWidget(self.save_path_button)
        autolabel_layout.addLayout(save_path_layout)

        # Thresholds
        threshold_layout = QHBoxLayout()
        self.conf_label = QLabel('Confidence Threshold:')
        self.conf_input = QLineEdit("0.25")
        self.iou_label = QLabel('IoU Threshold:')
        self.iou_input = QLineEdit("0.7")
        threshold_layout.addWidget(self.conf_label)
        threshold_layout.addWidget(self.conf_input)
        threshold_layout.addWidget(self.iou_label)
        threshold_layout.addWidget(self.iou_input)
        autolabel_layout.addLayout(threshold_layout)

        # Buttons
        autolabel_buttons_layout = QHBoxLayout()
        self.start_label_button = QPushButton('Start Auto-Labeling')
        self.start_label_button.clicked.connect(self.start_auto_labeling)
        self.open_labelimg_button = QPushButton('Review in labelImg')
        self.open_labelimg_button.clicked.connect(self.open_labelimg)
        self.open_labelimg_button.setEnabled(False) # Initially disabled
        autolabel_buttons_layout.addWidget(self.start_label_button)
        autolabel_buttons_layout.addWidget(self.open_labelimg_button)
        autolabel_layout.addLayout(autolabel_buttons_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        autolabel_layout.addWidget(self.progress_bar)
        
        self.autolabel_log = QTextEdit()
        self.autolabel_log.setReadOnly(True)
        autolabel_layout.addWidget(self.autolabel_log)

        autolabel_tab.setLayout(autolabel_layout)

        tabs.addTab(train_tab, "Training")
        tabs.addTab(autolabel_tab, "Auto-Labeling")

        main_layout.addWidget(tabs)
        self.setLayout(main_layout)

    def log_handler(self, message):
        self.log_area.append(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + '\n')
            except Exception as e:
                self.log_area.append(f"Failed to write to log file: {e}")

    def autolabel_log_handler(self, message):
        self.autolabel_log.append(message)
        if self.log_file: # Use the same log file if a process is running
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"[AutoLabel] {message}\n")
            except Exception as e:
                self.autolabel_log.append(f"Failed to write to log file: {e}")

    def populate_model_sizes(self):
        self.model_size_combo.clear()
        try:
            files = os.listdir('training_options')
            sizes = set()
            for f in files:
                if f.startswith('yolov8') and f.endswith('.pt'):
                    size = f.replace('yolov8', '').replace('.pt', '')
                    if size in ['n', 's', 'm', 'l', 'x']:
                        sizes.add(size)
            if sizes:
                self.model_size_combo.addItems(sorted(list(sizes)))
        except FileNotFoundError:
            self.log_handler("Error: 'training_options' directory not found.")

    def populate_activations(self):
        self.activation_combo.clear()
        try:
            files = os.listdir('training_options')
            activations = set()
            for f in files:
                if f.startswith('yolov8') and f.endswith('.yaml') and '-' in f:
                    activation = f.split('-')[-1].replace('.yaml', '')
                    activations.add(activation.capitalize())
            if any(f.startswith('yolov8') and f.endswith('.yaml') and '-' not in f for f in files):
                activations.add('Silu') # Add SiLU as a default activation

            if activations:
                self.activation_combo.addItems(sorted(list(activations)))
        except FileNotFoundError:
            self.log_handler("Error: 'training_options' directory not found.")

    def populate_device_options(self):
        """Populate device combo box with available GPU options"""
        self.device_combo.clear()
        options = []

        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()

                # Add auto option
                options.append(f"Auto (All {gpu_count} GPUs)")

                # Add individual GPU options
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    options.append(f"GPU {i}: {gpu_name[:30]}")  # Truncate long names

                # Add multi-GPU presets
                if gpu_count >= 2:
                    options.append(f"All GPUs (0-{gpu_count-1})")
                if gpu_count >= 4:
                    options.append("4 GPUs (0,1,2,3)")

                # Add CPU option
                options.append("CPU")

                # Add custom option
                options.append("Custom...")

            else:
                options.append("CPU (No GPU detected)")

        except Exception as e:
            options.append("CPU (Error detecting GPU)")

        self.device_combo.addItems(options)

    def on_device_changed(self, text):
        """Handle device combo box changes"""
        if text.startswith("Custom"):
            self.device_custom_input.setVisible(True)
        else:
            self.device_custom_input.setVisible(False)

    def get_device_string(self):
        """Get the actual device string to pass to training"""
        selected = self.device_combo.currentText()

        if selected.startswith("Custom"):
            return self.device_custom_input.text() if self.device_custom_input.text() else "0"
        elif selected.startswith("Auto"):
            return "auto"
        elif selected.startswith("GPU "):
            # Extract GPU index (e.g., "GPU 0: NVIDIA RTX" -> "0")
            return selected.split(":")[0].replace("GPU ", "")
        elif selected.startswith("All GPUs"):
            # Extract range (e.g., "All GPUs (0-3)" -> "0,1,2,3")
            if "(" in selected:
                range_str = selected.split("(")[1].split(")")[0]
                if "-" in range_str:
                    start, end = range_str.split("-")
                    return ",".join(str(i) for i in range(int(start), int(end)+1))
            return "auto"
        elif "4 GPUs" in selected:
            return "0,1,2,3"
        elif selected.startswith("CPU"):
            return "cpu"
        else:
            return "0"  # Default fallback

    def browse_dataset_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Dataset Root Folder')
        if path:
            self.dataset_path.setText(path)

    def browse_save_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Directory to Save Models')
        if path:
            self.save_dir_path.setText(path)

    def browse_classes_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Classes File', '', 'Text Files (*.txt)')
        if path:
            self.classes_file_path.setText(path)

    def browse_prep_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Directory for Prepared Dataset')
        if path:
            self.prep_output_path.setText(path)

    def browse_images(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Images Directory')
        if path:
            self.images_path.setText(path)

    def browse_model_label(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Model', '', 'PyTorch Models (*.pt)')
        if path:
            self.model_path_label.setText(path)

    def browse_save_path(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Directory to Save Annotations')
        if path:
            self.save_path_input.setText(path)

    def start_training(self):
        dataset_folder = self.dataset_path.text()
        classes_file = self.classes_file_path.text()
        val_ratio = self.val_ratio_input.text()
        prep_output_dir = self.prep_output_path.text()  # NEW: Get prepared dataset output directory
        save_dir = self.save_dir_path.text()
        model_size = self.model_size_combo.currentText()
        activation = self.activation_combo.currentText()
        epochs = self.epochs_input.text()
        device = self.get_device_string()  # Use new method
        lr0 = self.lr_input.text()
        dropout = self.dropout_input.text()
        batch_size = self.batch_size_input.text()

        if not all([dataset_folder, classes_file, val_ratio, save_dir, model_size, activation, epochs, device, lr0, dropout, batch_size]):
            self.log_handler("Please provide all required fields.")
            return

        # Create a new log file for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"log_{timestamp}.txt"
        self.log_handler(f"Starting training run. Log file: {self.log_file}")

        if prep_output_dir:
            self.log_handler(f"Prepared dataset will be saved to: {prep_output_dir}")
        else:
            self.log_handler("Using default prepared dataset directory")

        self.start_training_button.setEnabled(False)
        self.prep_progress_bar.setValue(0)
        self.training_thread = TrainingThread(
            dataset_folder, classes_file, val_ratio, save_dir, model_size,
            activation, epochs, device, lr0, dropout, batch_size,
            prep_output_dir if prep_output_dir else None  # Pass output dir
        )
        self.training_thread.log_signal.connect(self.log_handler)
        self.training_thread.prep_progress_signal.connect(self.prep_progress_bar.setValue)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

    def training_finished(self):
        self.start_training_button.setEnabled(True)
        self.log_handler("Training run finished.")
        self.log_file = None # Reset log file

    def start_auto_labeling(self):
        images_path = self.images_path.text()
        model_path = self.model_path_label.text()
        conf = self.conf_input.text()
        iou = self.iou_input.text()
        save_path = self.save_path_input.text()

        if not images_path or not model_path:
            self.autolabel_log_handler("Please provide all auto-labeling parameters.")
            return
        
        # If save path is not provided, use the image path as default
        if not save_path:
            save_path = images_path

        # Create a new log file for this run if not already logging
        if not self.log_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = f"log_{timestamp}.txt"
        self.autolabel_log_handler(f"Starting auto-labeling run. Log file: {self.log_file}")

        self.start_label_button.setEnabled(False)
        self.open_labelimg_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.autolabel_thread = AutoLabelThread(images_path, model_path, conf, iou, save_path)
        self.autolabel_thread.progress_signal.connect(self.progress_bar.setValue)
        self.autolabel_thread.log_signal.connect(self.autolabel_log_handler)
        self.autolabel_thread.finished.connect(self.autolabeling_finished)
        self.autolabel_thread.start()

    def autolabeling_finished(self):
        self.start_label_button.setEnabled(True)
        self.open_labelimg_button.setEnabled(True)
        self.autolabel_log_handler("Auto-labeling run finished.")
        self.log_file = None # Reset log file

    def open_labelimg(self):
        images_path = self.images_path.text()
        if not images_path:
            self.autolabel_log_handler("Image path is not set.")
            return
        
        try:
            # Assuming labelimg.exe is in the same directory or in PATH
            # and it can take the image directory as an argument.
            self.autolabel_log_handler(f"Opening {images_path} in labelImg...")
            subprocess.Popen(["labelimg.exe", images_path])
        except FileNotFoundError:
            self.autolabel_log_handler("Error: labelimg.exe not found. Make sure it is in the application directory.")
        except Exception as e:
            self.autolabel_log_handler(f"Failed to open labelImg: {e}")

    def save_settings(self):
        settings = {
            'dataset_path': self.dataset_path.text(),
            'classes_file_path': self.classes_file_path.text(),
            'val_ratio': self.val_ratio_input.text(),
            'prep_output_path': self.prep_output_path.text(),  # NEW
            'save_dir': self.save_dir_path.text(),
            'model_size': self.model_size_combo.currentText(),
            'activation': self.activation_combo.currentText(),
            'lr0': self.lr_input.text(),
            'dropout': self.dropout_input.text(),
            'epochs': self.epochs_input.text(),
            'device': self.device_combo.currentText(),  # Save combo selection
            'device_custom': self.device_custom_input.text(),
            'batch_size': self.batch_size_input.text(),
            'autolabel_images_path': self.images_path.text(),
            'autolabel_model_path': self.model_path_label.text(),
            'autolabel_save_path': self.save_path_input.text(),
            'autolabel_conf': self.conf_input.text(),
            'autolabel_iou': self.iou_input.text(),
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, 'r') as f:
                settings = json.load(f)

            self.dataset_path.setText(settings.get('dataset_path', ''))
            self.classes_file_path.setText(settings.get('classes_file_path', ''))
            self.val_ratio_input.setText(settings.get('val_ratio', '20'))
            self.prep_output_path.setText(settings.get('prep_output_path', ''))  # NEW
            self.save_dir_path.setText(settings.get('save_dir', 'trained_models'))
            self.model_size_combo.setCurrentText(settings.get('model_size', 'n'))
            self.activation_combo.setCurrentText(settings.get('activation', 'SiLU'))
            self.lr_input.setText(settings.get('lr0', '0.01'))
            self.dropout_input.setText(settings.get('dropout', '0.0'))
            self.epochs_input.setText(settings.get('epochs', '30'))

            # Load device settings
            device_setting = settings.get('device', 'GPU 0')
            if device_setting in [self.device_combo.itemText(i) for i in range(self.device_combo.count())]:
                self.device_combo.setCurrentText(device_setting)
            self.device_custom_input.setText(settings.get('device_custom', ''))

            self.batch_size_input.setText(settings.get('batch_size', '16'))
            self.images_path.setText(settings.get('autolabel_images_path', ''))
            self.model_path_label.setText(settings.get('autolabel_model_path', ''))
            self.save_path_input.setText(settings.get('autolabel_save_path', ''))
            self.conf_input.setText(settings.get('autolabel_conf', '0.25'))
            self.iou_input.setText(settings.get('autolabel_iou', '0.7'))

        except Exception as e:
            print(f"Error loading settings: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = YOLOv8TrainerUI()
    ex.show()
    sys.exit(app.exec_())
