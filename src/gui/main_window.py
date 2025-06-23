import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QMenu, QMenuBar,
                             QMessageBox, QProgressBar, QPushButton,
                             QStatusBar, QTextEdit, QVBoxLayout, QWidget)

from src.core import FileManager, ModelManager, SettingsManager, get_logger
from src.gui.about_dialog import AboutDialog
from src.gui.image_mask_viewer import ImageMaskViewer
from src.gui.models_window import DeepLearningModel, ModelsWindow
from src.gui.settings_dialog import SettingsDialog

# Try to import torchstain for stain normalization
try:
    import torchstain
except ImportError:
    torchstain = None


class MainWindow(QMainWindow):
    """Main application window."""

    # Define signals
    status_signal = pyqtSignal(str)

    def __init__(self, db_session=None):
        super().__init__()
        self.setWindowTitle("Deep Histo Image AI")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize logger
        self.logger = get_logger(__name__)

        # Initialize managers
        self.settings_manager = SettingsManager()

        # Get paths from settings
        input_path = self.settings_manager.get("Paths", "input_path", "input")
        output_path = self.settings_manager.get(
            "Paths", "output_path", "output")

        # Initialize file manager with paths
        self.file_manager = FileManager(input_path, output_path)

        self.model_manager = None
        self.db_session = db_session

        # Initialize prediction-related attributes
        self.prediction_worker = None
        self.prediction_start_time = None
        self.prediction_timer = QTimer()
        self.prediction_timer.timeout.connect(self._update_remaining_time)

        # Set up UI
        self._setup_ui()

        # Set up database and model manager
        self._setup_database()

        # Create required directories
        self._create_required_directories()

        # Load models
        self._load_models()

        # Log startup
        self._log_status("Application started", is_prediction_status=False)

        # Connect status signal
        self.status_signal.connect(
            lambda msg: self._log_status(
                msg, is_prediction_status=True))

    def _setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create left and right panels
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # Create menu bar
        self._create_menu_bar()

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

    def _setup_database(self):
        # Initialize model manager
        models_path = self.settings_manager.get_path(
            "Paths", "models_path", "models")
        self.model_manager = ModelManager(self.db_session, models_path)

    def _create_left_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model selection section
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentTextChanged.connect(
            self._on_model_selection_change)

        # Buttons
        button_layout = QHBoxLayout()
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self._start_prediction)
        self.view_button = QPushButton("View Results")
        self.view_button.clicked.connect(self._view_results)
        button_layout.addWidget(self.predict_button)
        button_layout.addWidget(self.view_button)

        # Add widgets to layout
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addLayout(button_layout)
        layout.addStretch()

        return widget

    def _create_right_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Progress section
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 1px solid #AAAAAA;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """
        )
        self.time_label = QLabel("Time remaining: --:--:--")

        # Status log
        status_label = QLabel("Status Log:")
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)

        # Add widgets to layout
        layout.addWidget(progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.time_label)
        layout.addWidget(status_label)
        layout.addWidget(self.status_text)

        return widget

    def _create_menu_bar(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        import_model_action = file_menu.addAction("Import model from file")
        import_model_action.triggered.connect(self._open_model_import_dialog)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        # Models menu
        models_menu = menubar.addMenu("Models")

        models_menu.addSeparator()

        models_action = models_menu.addAction("Show available models")
        models_action.triggered.connect(self._open_models_window)

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        settings_action = settings_menu.addAction("Settings")
        settings_action.triggered.connect(self._open_settings_dialog)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self._show_about)

    def _browse_folder(self, folder_type):
        folder = QFileDialog.getExistingDirectory(
            self, f"Select {folder_type.capitalize()} Folder", os.path.expanduser("~"))
        if folder:
            if folder_type == "input":
                self.input_path.setText(folder)
                # Save to settings
                self.settings_manager.set("Paths", "input_path", folder)
                self.settings_manager.save_settings()
                # Update file manager
                self.file_manager.input_path = folder
            else:
                self.output_path.setText(folder)
                # Save to settings
                self.settings_manager.set("Paths", "output_path", folder)
                self.settings_manager.save_settings()
                # Update file manager
                self.file_manager.output_path = folder

            # Ensure directory exists
            os.makedirs(folder, exist_ok=True)

    def _start_prediction(self):
        """Start the prediction process."""
        # Check if prediction is already running
        if (
            hasattr(self, "prediction_worker")
            and self.prediction_worker is not None
            and self.prediction_worker.isRunning()
        ):
            self._cancel_prediction()
            return

        # Check if model is selected
        if self.model_combo.count() == 0 or self.model_combo.currentIndex() == -1:
            QMessageBox.warning(
                self,
                "No Model Selected",
                "Please select a model first.")
            return

        # Get selected model GUID
        selected_model_guid = self.model_combo.currentData()
        self._log_status(
            f"Selected model GUID: {selected_model_guid}",
            is_prediction_status=True)

        # Verify model is loaded
        if not self.model_manager.current_model:
            # Try to load the model
            self._log_status(f"Loading model...", is_prediction_status=True)
            success, message = self.model_manager.load_model(
                selected_model_guid)
            if not success:
                self._log_status(
                    f"Error loading model: {message}",
                    is_prediction_status=True)
                QMessageBox.warning(
                    self,
                    "Model Loading Error",
                    f"Failed to load the selected model: {message}",
                )
                return

        self._log_status(
            f"Model loaded successfully",
            is_prediction_status=True)

        # Get paths
        input_path = self.settings_manager.get("Paths", "input_path", "input")
        output_path = self.settings_manager.get(
            "Paths", "output_path", "output")

        if not input_path or not os.path.isdir(input_path):
            QMessageBox.warning(
                self,
                "Invalid Input Path",
                "Please set a valid input directory in the settings.",
            )
            return

        if not output_path or not os.path.isdir(output_path):
            QMessageBox.warning(
                self,
                "Invalid Output Path",
                "Please set a valid output directory in the settings.",
            )
            return

        # Get image files
        image_files = [
            f
            for f in os.listdir(input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]

        if not image_files:
            QMessageBox.warning(
                self,
                "No Images Found",
                f"No supported image files found in {input_path}.",
            )
            return

        # Setup UI for prediction
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 0)  # Show busy indicator
        self.progress_bar.setFormat("Initializing...")
        self.prediction_start_time = time.time()
        self.time_label.setText("Time remaining: calculating...")
        self.prediction_timer.start(1000)  # Update every second

        # Disable UI elements during prediction
        self.predict_button.setText("Cancel")
        self.predict_button.clicked.disconnect()
        self.predict_button.clicked.connect(self._cancel_prediction)
        self.model_combo.setEnabled(False)
        self.view_button.setEnabled(False)

        # Log start of prediction
        self._log_status(
            f"Starting prediction with model: {self.model_combo.currentText()}",
            is_prediction_status=True,
        )
        self._log_status(
            f"Processing {len(image_files)} images from {input_path}",
            is_prediction_status=True,
        )

        # Get device from settings
        device = self.settings_manager.get("Default", "device", "cpu")
        model_guid = self.model_combo.currentData()
        if model_guid:
            self.model_manager.device = device
            self.model_manager.load_model(model_guid) 

        # Create and start the worker
        # self.prediction_worker = PredictionWorker(
        #     model_guid=self.model_combo.currentData(),
        #     input_path=input_path,
        #     output_path=output_path,
        #     image_files=image_files,
        #     status_signal=self.status_signal,
        #     progress_signal=self._update_progress,
        #     device=device,
        #     model_manager=self.model_manager,
        # )

        self.prediction_worker = PredictionWorker(
            model_guid=model_guid,
            input_path=input_path,
            output_path=output_path,
            image_files=image_files,
            status_signal=self.status_signal,
            progress_signal=self._update_progress,
            device=device,
            model_manager=self.model_manager,
        )

        # Connect signals
        self.prediction_worker.finished.connect(self._on_prediction_finished)

        # Start the worker
        self.prediction_worker.start()

    def _cancel_prediction(self):
        """Cancel the running prediction process."""
        if (
            hasattr(self, "prediction_worker")
            and self.prediction_worker is not None
            and self.prediction_worker.isRunning()
        ):
            reply = QMessageBox.question(
                self,
                "Cancel Prediction",
                "Are you sure you want to cancel the prediction?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._log_status(
                    "Cancelling prediction...",
                    is_prediction_status=True)
                self.prediction_worker.requestInterruption()

    def _on_prediction_finished(self):
        """Handle prediction completion."""
        # Stop the timer
        self.prediction_timer.stop()

        # Reset UI
        self.predict_button.setText("Predict")
        self.predict_button.clicked.disconnect()
        self.predict_button.clicked.connect(self._start_prediction)
        self.model_combo.setEnabled(True)
        self.view_button.setEnabled(True)

        # Reset progress bar
        self.progress_bar.setRange(0, 100)
        if (
            hasattr(self, "prediction_worker")
            and self.prediction_worker is not None
            and self.prediction_worker.isInterruptionRequested()
        ):
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Cancelled")
            self.time_label.setText("Time remaining: --:--:--")
            # Log cancellation
            self._log_status("Prediction cancelled", is_prediction_status=True)
        else:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Complete (100%)")

            # Calculate total time
            if self.prediction_start_time is not None:
                total_time = time.time() - self.prediction_start_time
                time_str = str(timedelta(seconds=int(total_time)))
                self.time_label.setText(f"Total time: {time_str}")

            # Log completion
            self._log_status("Prediction completed", is_prediction_status=True)

            # Show message to user
            QMessageBox.information(
                self,
                "Prediction Complete",
                "Prediction completed successfully")

    def _update_remaining_time(self):
        """Update the remaining time display."""
        if self.prediction_start_time is None:
            return

        elapsed = time.time() - self.prediction_start_time
        progress = self.progress_bar.value()

        if progress > 0 and progress < 100:
            # Calculate estimated total time based on current progress
            total_estimated = elapsed * 100 / progress
            remaining = total_estimated - elapsed

            # Format the remaining time nicely
            if remaining < 60:
                remaining_time = f"{int(remaining)} seconds"
            elif remaining < 3600:
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                remaining_time = f"{minutes}:{seconds:02d} minutes"
            else:
                hours = int(remaining // 3600)
                minutes = int((remaining % 3600) // 60)
                remaining_time = f"{hours}:{minutes:02d} hours"

            self.time_label.setText(f"Time remaining: {remaining_time}")

            # Update status bar with more detailed information
            elapsed_formatted = str(timedelta(seconds=int(elapsed)))
            self.statusBar.showMessage(
                f"Elapsed: {elapsed_formatted} | Progress: {progress}% | Est. remaining: {remaining_time}"
            )
        elif progress == 100:
            # Show total time when complete
            total_time = str(timedelta(seconds=int(elapsed)))
            self.time_label.setText(f"Total time: {total_time}")
            self.statusBar.showMessage(f"Completed in {total_time}")
        else:
            self.time_label.setText("Time remaining: calculating...")
            self.statusBar.showMessage("Initializing prediction...")

    def _log_status(self, message, is_prediction_status=False):
        """
        Log a status message.

        Args:
            message: The message to log
            is_prediction_status: If True, the message will be displayed in the status text field.
                                 Otherwise, it will only be logged to the log file.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Always log to file, with a prefix for prediction-related messages
        if is_prediction_status:
            self.logger.info(f"[PREDICTION] {message}")
        else:
            self.logger.info(message)

        # Only show prediction-related messages in the status text field
        if is_prediction_status:
            self.status_text.append(f"[{timestamp}] {message}")

        # Always update the status bar with the latest message
        self.statusBar.showMessage(f"{message}", 5000)  # Show for 5 seconds

    def _on_model_selection_change(self, model_name):
        """Handle model selection change."""
        try:
            # Get the selected model GUID
            selected_model_guid = self.model_combo.currentData()
            if not selected_model_guid:
                self._log_status(
                    "No model selected",
                    is_prediction_status=False)
                return

            self._log_status(
                f"Loading model: {model_name}",
                is_prediction_status=False)

            # Load the model
            success, message = self.model_manager.load_model(
                selected_model_guid)
            if success:
                self._log_status(
                    f"Model loaded successfully: {model_name}",
                    is_prediction_status=False,
                )
            else:
                self._log_status(
                    f"Error loading model: {message}",
                    is_prediction_status=False)

        except Exception as e:
            self._log_status(
                f"Error during model selection: {str(e)}",
                is_prediction_status=False)
            import traceback

            self._log_status(
                traceback.format_exc(),
                is_prediction_status=False)

    def _open_models_window(self):
        self._log_status(
            "Opening Models window...",
            is_prediction_status=False)
        models_window = ModelsWindow(self, self.model_manager)
        models_window.show()

    def _open_settings_dialog(self):
        """Show the settings dialog and apply changes if accepted."""
        self._log_status(
            "Showing Settings dialog...",
            is_prediction_status=False)

        # Create the settings dialog with current settings
        dialog = SettingsDialog(self)
        result = dialog.exec()

        if result == SettingsDialog.DialogCode.Accepted:
            # Save the settings
            dialog.save_settings()

            # Reload settings in the settings manager
            self.settings_manager.load_settings()

            # Update UI based on new settings
            self.statusBar.showMessage("Settings updated", 3000)
        else:
            self._log_status(
                "Settings dialog cancelled",
                is_prediction_status=False)

    def _apply_settings(self):
        """Apply settings from the settings manager to the UI."""
        # Get paths from settings
        input_path = self.settings_manager.get(
            "Paths", "input_path", "input_folder")
        output_path = self.settings_manager.get(
            "Paths", "output_path", "output_folder")
        models_path = self.settings_manager.get(
            "Paths", "models_path", "models")

        # Ensure directories exist
        os.makedirs(input_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        # Update UI
        self.input_path.setText(input_path)
        self.output_path.setText(output_path)

        # Initialize or update file and model managers
        self.file_manager = FileManager(input_path, output_path)
        # self.model_manager = ModelManager(self.db_session, models_path) 

        device = self.settings_manager.get("Default", "device", "cpu")
        self.model_manager = ModelManager(self.db_session, models_path, device=device)
        current_index = self.model_combo.currentIndex()
        if current_index >= 0:
            model_guid = self.model_combo.currentData()
            if model_guid:
                self.model_manager.load_model(model_guid)

        # Apply other settings as needed
        # For example, you might apply theme changes, language changes, etc.

    def _show_about(self):
        """Show the about dialog."""

        self._log_status("Showing About dialog...", is_prediction_status=False)

        dialog = AboutDialog(self)
        dialog.exec()

    def _open_model_import_dialog(self):
        """Open the model import dialog."""
        from .model_import_dialog import ModelImportDialog

        dialog = ModelImportDialog(self)
        result = dialog.exec()

        if result == ModelImportDialog.DialogCode.Accepted:
            file_path = dialog.get_file_path()
            self._log_status(
                f"Model import accepted: {file_path}",
                is_prediction_status=False)
            self._import_model(Path(file_path))
        else:
            self._log_status(
                "Model import cancelled",
                is_prediction_status=False)

    def _import_model(self, archive_path: Path):
        """Import a model from an archive file."""

        try:
            # Create temporary directory for extraction
            temp_dir = Path("temp/import")

            # Extract archive to read description.json first
            import json
            import zipfile

            # Clear and create temp directory
            if temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract archive
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Check if description.json exists
            description_file = next((file for file in temp_dir.glob(
                "**/*") if file.name == "description.json"), None, )
            if not description_file:
                raise ValueError("description.json not found in archive")

            # Read description.json to get model_guid if it exists
            with open(description_file, "r") as f:
                description_data = json.load(f)

            # Use model_guid from description.json if it exists, otherwise
            # generate a new one
            if "model_guid" in description_data and description_data["model_guid"]:
                model_guid = description_data["model_guid"]
                self._log_status(
                    f"Using model_guid from description.json: {model_guid}",
                    is_prediction_status=False,
                )
            else:
                # Someting wrong with GUID in the description.json
                # TODO remove this
                # model_guid = str(uuid.uuid4())
                self._log_status(
                    f"Someting wrong with GUID in the description.json",
                    is_prediction_status=False,
                )

            # Check if a model with this GUID already exists
            model_dir = Path("models") / model_guid
            if model_dir.exists():
                self._log_status(
                    f"Model with GUID {model_guid} already exists",
                    is_prediction_status=False,
                )

                # Read the existing model's description.json
                existing_description_file = model_dir / "description.json"
                if existing_description_file.exists():
                    try:
                        with open(existing_description_file, "r") as f:
                            existing_description_data = json.load(f)

                        # Prepare comparison text
                        comparison_text = (
                            "<html><body><h3>Model with the same GUID already exists</h3>"
                        )
                        comparison_text += (
                            "<table border='1' cellpadding='5' style='border-collapse: collapse;'>"
                        )
                        comparison_text += (
                            "<tr><th>Field</th><th>Existing Model</th><th>New Model</th></tr>"
                        )

                        # Fields to compare
                        fields_to_compare = [
                            "model_name",
                            "model_architecture",
                            "model_encoder",
                            "model_in_channels",
                            "model_num_classes",
                            "model_description",
                            "model_version",
                            "create_date",
                        ]

                        for field in fields_to_compare:
                            existing_value = existing_description_data.get(
                                field, "N/A")
                            new_value = description_data.get(field, "N/A")

                            # Highlight differences
                            row_style = ""
                            if existing_value != new_value:
                                row_style = " style='background-color: #FFEEEE;'"

                            comparison_text += f"<tr{row_style}><td><b>{field}</b></td><td>{existing_value}</td><td>{new_value}</td></tr>"

                        comparison_text += "</table><p>Do you want to replace the existing model with the new one?</p></body></html>"

                        # Show comparison dialog
                        message_box = QMessageBox(self)
                        message_box.setWindowTitle("Model Already Exists")
                        message_box.setText(comparison_text)
                        message_box.setTextFormat(Qt.TextFormat.RichText)
                        message_box.setStandardButtons(
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        message_box.setDefaultButton(
                            QMessageBox.StandardButton.No)

                        # Adjust size to fit content
                        message_box.setMinimumWidth(600)

                        # Get user decision
                        result = message_box.exec()

                        if result == QMessageBox.StandardButton.Yes:
                            # User chose to replace the existing model
                            self._log_status(
                                f"Replacing existing model with GUID {model_guid}",
                                is_prediction_status=False,
                            )

                            # Remove existing model directory
                            import shutil

                            shutil.rmtree(model_dir)
                        else:
                            # User chose to keep the existing model
                            self._log_status(
                                f"Import cancelled, keeping existing model with GUID {model_guid}",
                                is_prediction_status=False,
                            )
                            return
                    except Exception as e:
                        self._log_status(
                            f"Error reading existing model description: {str(e)}",
                            is_prediction_status=False,
                        )

            # Create model directory
            model_dir.mkdir(parents=True, exist_ok=True)

            success, message = self.file_manager.import_model_files(
                archive_path=archive_path, temp_dir=temp_dir, model_dir=model_dir)

            if not success:
                raise ValueError(message)

            success, message = self.model_manager.validate_model_compatibility(
                model_dir / "description.json"
            )

            if not success:
                raise ValueError(message)

            # Save model to database
            self._save_imported_model_to_db(model_guid, model_dir)
            self._log_status(
                "Model imported successfully",
                is_prediction_status=False)
            QMessageBox.information(
                self,
                "Success",
                "Model imported successfully.\nYou can now select it from the model list.",
            )

        except Exception as e:
            self._log_status(
                f"Error importing model: {str(e)}",
                is_prediction_status=False)
            QMessageBox.critical(
                self, "Error", f"Failed to import model:\n{str(e)}")

    def _save_imported_model_to_db(self, model_guid: str, model_dir: Path):
        """
        Save imported model details to the database.

        Args:
            model_guid: Model GUID
            model_dir: Path to model directory
        """
        try:
            self._log_status(
                f"Saving model {model_guid} to database...",
                is_prediction_status=False)

            # Read description.json
            description_file = model_dir / "description.json"
            if not description_file.exists():
                raise ValueError(f"description.json not found in {model_dir}")

            with open(description_file, "r") as f:
                description_data = json.load(f)

            # Find model file (*.pt or *.pth)
            model_files = list(model_dir.glob("*.pt")) + \
                list(model_dir.glob("*.pth"))
            if not model_files:
                raise ValueError(
                    f"No model file (*.pt or *.pth) found in {model_dir}")

            model_file_name = model_files[0].name

            # Get current datetime for last_modified
            current_datetime = datetime.utcnow()

            # Extract all fields from description.json with default values if
            # not present
            model_name = description_data.get("model_name", "Unnamed Model")
            model_description = description_data.get("model_description", "")
            model_architecture = description_data.get(
                "model_architecture", "Unet")
            model_encoder = description_data.get("model_encoder", "resnet34")
            model_weights = description_data.get("model_weights", "imagenet")
            model_in_channels = description_data.get("model_in_channels", 3)
            model_num_classes = description_data.get("model_num_classes", 2)
            model_class_names = description_data.get("model_class_names", "")
            model_class_colours = description_data.get(
                "model_class_colours", "")
            staining = description_data.get("staining", "none")
            image_size = description_data.get("image_size", 512)
            normalization_method = description_data.get(
                "normalization_method", "none")
            stain_normalization_method = description_data.get(
                "stain_normalization_method")
            training_image_example = description_data.get(
                "training_image_example")
            training_image_mask_example = description_data.get(
                "training_image_mask_example")
            normalization_image = description_data.get("normalization_image")
            model_version = description_data.get("model_version", "1.0.0")

            # Use create_date from description.json if available, otherwise use
            # current datetime
            create_date_str = description_data.get("create_date")
            if create_date_str:
                try:
                    # Try to parse the date string from description.json
                    # Assuming ISO format (YYYY-MM-DDTHH:MM:SS)
                    create_date = datetime.fromisoformat(create_date_str)
                except (ValueError, TypeError):
                    # If parsing fails, use current datetime
                    create_date = current_datetime
            else:
                create_date = current_datetime

            # Register model in database
            success, message = self.model_manager.register_model(
                model_guid=model_guid,
                model_name=model_name,
                model_file_name=model_file_name,
                model_description=model_description,
                model_architecture=model_architecture,
                model_encoder=model_encoder,
                model_weights=model_weights,
                model_in_channels=model_in_channels,
                model_num_classes=model_num_classes,
                model_class_names=model_class_names,
                model_class_colours=model_class_colours,
                staining=staining,
                image_size=image_size,
                normalization_method=normalization_method,
                stain_normalization_method=stain_normalization_method,
                training_image_example=training_image_example,
                training_image_mask_example=training_image_mask_example,
                normalization_image=normalization_image,
                model_version=model_version,
                create_date=create_date,
                last_modified=current_datetime,
            )

            if not success:
                # If model already exists in database, update it
                self._log_status(
                    f"Model already exists in database, updating: {message}",
                    is_prediction_status=False,
                )

                # Get existing model from database
                existing_model = (
                    self.db_session.query(DeepLearningModel)
                    .filter_by(model_guid=model_guid)
                    .first()
                )

                if existing_model:
                    # Update all model fields
                    existing_model.model_name = model_name
                    existing_model.model_file_name = model_file_name
                    existing_model.model_description = model_description
                    existing_model.model_architecture = model_architecture
                    existing_model.model_encoder = model_encoder
                    existing_model.model_weights = model_weights
                    existing_model.model_in_channels = model_in_channels
                    existing_model.model_num_classes = model_num_classes
                    existing_model.model_class_names = model_class_names
                    existing_model.model_class_colours = model_class_colours
                    existing_model.staining = staining
                    existing_model.image_size = image_size
                    existing_model.normalization_method = normalization_method
                    existing_model.stain_normalization_method = stain_normalization_method
                    existing_model.training_image_example = training_image_example
                    existing_model.training_image_mask_example = training_image_mask_example
                    existing_model.normalization_image = normalization_image
                    existing_model.model_version = model_version
                    # Don't update create_date, only last_modified
                    existing_model.last_modified = current_datetime

                    # Commit changes
                    self.db_session.commit()
                    self._log_status(
                        f"Model {model_guid} updated in database",
                        is_prediction_status=False,
                    )
                else:
                    raise ValueError(
                        f"Model {model_guid} not found in database but register_model reported it exists"
                    )
            else:
                self._log_status(
                    f"Model {model_guid} registered in database: {message}",
                    is_prediction_status=False,
                )

            # Reload models list
            self._load_models()

        except Exception as e:
            self._log_status(
                f"Error saving model to database: {str(e)}",
                is_prediction_status=False)
            raise

    def _load_models(self):
        """Load available models into the combo box."""
        try:
            self._log_status("Loading models...", is_prediction_status=False)

            # Clear existing items
            self.model_combo.clear()

            # Get models from model manager
            models = self.model_manager.get_all_models()

            # Add models to combo box
            if models:
                for model in models:
                    self.model_combo.addItem(
                        f"{model.model_name} ({model.model_architecture})",
                        model.model_guid,
                    )

                # Select first model
                self.model_combo.setCurrentIndex(0)
                self._log_status(
                    f"Loaded {len(models)} models",
                    is_prediction_status=False)

        except Exception as e:
            self._log_status(
                f"Error loading models: {str(e)}",
                is_prediction_status=False)
            import traceback

            self._log_status(
                traceback.format_exc(),
                is_prediction_status=False)

    def _create_required_directories(self):
        """Create all required directories based on settings."""
        try:
            # Get paths from settings and create directories
            paths = [
                self.settings_manager.get_path("Paths", "input_path", "input_folder"),
                self.settings_manager.get_path("Paths", "output_path", "output_folder"),
                self.settings_manager.get_path("Paths", "models_path", "models"),
                self.settings_manager.get_path("Paths", "config_path", "config"),
                self.settings_manager.get_path("Paths", "log_path", "logs"),
                self.settings_manager.get_path("Paths", "data_path", "data"),
                self.settings_manager.get_path("Paths", "docs_path", "docs"),
                self.settings_manager.get_path("Paths", "db_migrations_path", "src/migrations"),
                self.settings_manager.get_path("Paths", "tests_path", "src/tests"),
            ]

            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
                    self._log_status(
                        f"Created directory: {path}",
                        is_prediction_status=False)

        except Exception as e:
            self._log_status(
                f"Error creating directories: {str(e)}",
                is_prediction_status=False)
            import traceback

            self._log_status(
                traceback.format_exc(),
                is_prediction_status=False)
            QMessageBox.warning(
                self,
                "Directory Error",
                f"There was an error creating required directories: {str(e)}",
            )

    def _show_image_and_mask(self):
        """Show a dialog with input images and their corresponding masks."""
        from src.gui.image_mask_viewer import ImageMaskViewer

        # Get paths from settings
        input_path = self.settings_manager.get(
            "Paths", "input_path", "input_folder")
        output_path = self.settings_manager.get(
            "Paths", "output_path", "output_folder")

        # Log action
        self._log_status(
            f"Opening Image and Mask Viewer with input path: {input_path}, output path: {output_path}",
            is_prediction_status=False,
        )

        # Create and show the dialog
        viewer = ImageMaskViewer(
            self,
            input_path,
            output_path,
            self.model_manager)
        viewer.exec()

    def _view_results(self):
        """Open the image and mask viewer."""
        # Get input and output paths
        input_path = self.settings_manager.get("Paths", "input_path", "")
        output_path = self.settings_manager.get("Paths", "output_path", "")

        # Validate paths
        if not input_path or not os.path.isdir(input_path):
            QMessageBox.warning(
                self,
                "Invalid Input Path",
                "Please set a valid input directory in the settings.",
            )
            return

        if not output_path or not os.path.isdir(output_path):
            QMessageBox.warning(
                self,
                "Invalid Output Path",
                "Please set a valid output directory in the settings.",
            )
            return

        # Open viewer
        self._log_status(
            "Opening image and mask viewer",
            is_prediction_status=False)
        viewer = ImageMaskViewer(
            self,
            input_path,
            output_path,
            self.model_manager)
        viewer.exec()

    def _update_progress(self, value):
        """Update the progress bar with the given value."""
        # If progress bar is in busy state, set it back to normal range
        if self.progress_bar.minimum() == self.progress_bar.maximum():
            self.progress_bar.setRange(0, 100)

        # Update the progress value
        self.progress_bar.setValue(value)

        # Update the format to show current/total
        if hasattr(
                self,
                "prediction_worker") and hasattr(
                self.prediction_worker,
                "image_files"):
            total_images = len(self.prediction_worker.image_files)
            current_image = int(value * total_images / 100)
            if current_image > total_images:
                current_image = total_images
            self.progress_bar.setFormat(
                f"{value}% ({current_image}/{total_images} images)")
        else:
            self.progress_bar.setFormat(f"{value}%")


# TODO move to core
class ImageSegmentationPredictor:
    def __init__(
        self,
        model,
        device,
        img_size,
        normalization_method="none",
        stain_normalization_method=None,
        target_image_path=None,
        mean=None,
        std=None,
        status_signal=None,
    ):
        self.model = model
        self.device = device
        # Debug: print device info
        print("Predictor device:", self.device)
        self.img_size = img_size
        self.normalization_method = normalization_method
        self.stain_normalization_method = stain_normalization_method
        self.mean = mean
        self.std = std
        self.normalize = None
        self.stain_normalizer = None
        self.status_signal = status_signal

        # Setup normalization
        if self.normalization_method == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        elif self.normalization_method == "custom":
            if self.mean is None or self.std is None:
                raise ValueError(
                    "For 'custom' normalization, mean and std must be provided.")
            self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        elif self.normalization_method == "stain":
            if stain_normalization_method is None or target_image_path is None:
                raise ValueError(
                    "For 'stain' normalization, stain_normalization_method and target_image_path must be provided."
                )

            if torchstain is not None:
                try:
                    target_image = cv2.imread(target_image_path)
                    if target_image is None:
                        self._log_message(
                            f"Warning: Could not load target image for stain normalization: {target_image_path}"
                        )
                    else:
                        target_image = cv2.cvtColor(
                            target_image, cv2.COLOR_BGR2RGB)

                        normalizers = torchstain.normalizers
                        if stain_normalization_method == "macenko":
                            self.stain_normalizer = normalizers.MacenkoNormalizer(
                                backend="numpy")
                        elif stain_normalization_method == "reinhard":
                            self.stain_normalizer = normalizers.ReinhardNormalizer(
                                backend="numpy", method=None)
                        elif stain_normalization_method == "reinhard_modified":
                            self.stain_normalizer = normalizers.ReinhardNormalizer(
                                backend="numpy", method="modified")
                        else:
                            raise ValueError(
                                f"Unknown stain normalization method: {stain_normalization_method}"
                            )

                        self.stain_normalizer.fit(target_image)
                except Exception as e:
                    self._log_message(
                        f"Warning: Error setting up stain normalization: {str(e)}")

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def _log_message(self, message):
        """Log a message using status_signal if available, otherwise print"""
        if self.status_signal:
            self.status_signal.emit(message)
        else:
            print(message)

    def preprocess_image(self, img_path):
        """Load and preprocess an image for prediction."""
        try:
            # Load and resize image
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
            image = cv2.resize(image, (self.img_size, self.img_size))

            # Apply stain normalization if needed
            if self.normalization_method == "stain" and self.stain_normalizer is not None:
                try:
                    normalized = self.stain_normalizer.normalize(image)
                    image = normalized[0] if isinstance(
                        normalized, tuple) else normalized
                except Exception as e:
                    self._log_message(
                        f"Warning: Stain normalization failed: {str(e)}")

            # Convert to tensor and normalize
            image = self.to_tensor(image)

            if self.normalize is not None:
                image = self.normalize(image)

            # Add batch dimension and move to device
            # return image.unsqueeze(0).to(self.device)
            # return image.unsqueeze(0).to(torch.device(self.device))

            # Add batch dimension and move to device
            tensor = image.unsqueeze(0).to(torch.device(self.device))
            
            self._log_message(f"preprocess_image: self.device = {self.device}")
            self._log_message(f"preprocess_image: torch.cuda.is_available() = {torch.cuda.is_available()}")
            self._log_message(f"preprocess_image: tensor.device = {tensor.device}")
            return tensor

        except Exception as e:
            self._log_message(
                f"Error preprocessing image {img_path}: {str(e)}")
            return None

    def predict_mask(self, image_tensor):
        """Generate prediction mask from image tensor."""
        # Debug: print device info
        self._log_message(f"Model device: {next(self.model.parameters()).device}")
        self._log_message(f"Input tensor device: {image_tensor.device}")
        assert next(self.model.parameters()).device == image_tensor.device, (
    f"Model device {next(self.model.parameters()).device} and tensor device {image_tensor.device} do not match"
)
        try:
            self.model.eval()
            with torch.no_grad():
                # Get model output
                logits = self.model(image_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]  # Some models return (output, features)

                # Get class predictions directly
                preds = torch.argmax(logits, dim=1)
                return preds.squeeze(0).cpu().numpy()

        except Exception as e:
            self._log_message(f"Error during prediction: {str(e)}")
            return None

    def save_mask(self, mask, save_path):
        """Save mask as a 16-bit TIFF file."""
        try:
            # Convert to 16-bit unsigned integer
            mask_16bit = mask.astype(np.uint16)
            mask_16bit_image = Image.fromarray(mask_16bit)
            mask_16bit_image.save(save_path, format="TIFF")
        except Exception as e:
            self._log_message(f"Error saving mask: {str(e)}")
            # Try to save with a different approach if the first one fails
            try:
                import imageio

                self._log_message("Trying alternative save method...")
                imageio.imwrite(save_path, mask_16bit)
            except Exception as e2:
                self._log_message(
                    f"Alternative save method also failed: {str(e2)}")


class PredictionWorker(QThread):
    # Define signals at class level
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(
        self,
        model_guid,
        input_path,
        output_path,
        image_files,
        status_signal,
        progress_signal,
        device,
        model_manager,
        parent=None,
    ):
        super().__init__(parent)
        self.model_guid = model_guid
        self.input_path = input_path
        self.output_path = output_path
        self.image_files = image_files
        # Connect external signals to our class signals
        self.status_signal.connect(status_signal)
        self.progress_signal.connect(progress_signal)
        self.device = device
        self.model_manager = model_manager
        self.predictor = None

    def run(self):
        """Run prediction on all images."""
        try:
            # Get model data
            if not self.model_manager.current_model:
                self.status_signal.emit(f"Error: No model loaded")
                return

            model_data = self.model_manager.current_model["metadata"]
            model = self.model_manager.current_model["model"]

            # Create predictor
            self.status_signal.emit("Creating predictor...")

            # Check for interruption
            if self.isInterruptionRequested():
                self.status_signal.emit("Prediction interrupted")
                return

            self.predictor = self._create_predictor(model, model_data)

            # Update progress to show initialization is complete
            self.progress_signal.emit(1)  # Small initial progress

            if self.predictor is None:
                self.status_signal.emit("Failed to create predictor")
                return

            # Calculate progress increment per image
            total_images = len(self.image_files)
            progress_per_image = 98 / total_images  # Reserve 1% for start and 1% for end

            # Process each image
            for i, img_name in enumerate(self.image_files):
                if self.isInterruptionRequested():
                    self.status_signal.emit("Prediction interrupted")
                    return

                img_path = os.path.join(self.input_path, img_name)
                save_path = os.path.join(
                    self.output_path, f"{os.path.splitext(img_name)[0]}.tif")

                # Update status
                self.status_signal.emit(
                    f"Processing image {i+1}/{len(self.image_files)}: {img_name}"
                )

                try:
                    # Update progress before processing (to show we're working
                    # on this image)
                    current_progress = 1 + int(i * progress_per_image)
                    self.progress_signal.emit(current_progress)

                    # Check for interruption
                    if self.isInterruptionRequested():
                        self.status_signal.emit("Prediction interrupted")
                        return

                    # Process image
                    success = self._process_image(img_path, save_path)
                    if not success:
                        self.status_signal.emit(f"Error processing {img_name}")
                        continue

                    # Check for interruption
                    if self.isInterruptionRequested():
                        self.status_signal.emit("Prediction interrupted")
                        return

                    # Update progress after processing
                    current_progress = 1 + int((i + 1) * progress_per_image)
                    self.progress_signal.emit(current_progress)

                except Exception as e:
                    self.status_signal.emit(
                        f"Error processing {img_name}: {str(e)}")
                    continue

            # Final progress update
            if not self.isInterruptionRequested():
                self.progress_signal.emit(99)  # Almost done
                self.status_signal.emit("Finalizing...")
                time.sleep(0.5)  # Small delay to show the finalizing state
                self.progress_signal.emit(100)  # Complete
                self.status_signal.emit("Prediction completed successfully")

        except Exception as e:
            self.status_signal.emit(f"Error during prediction: {str(e)}")
            import traceback

            self.status_signal.emit(f"Traceback: {traceback.format_exc()}")

    def _create_predictor(self, model, model_data):
        """Create image segmentation predictor."""
        try:
            self.status_signal.emit("Creating predictor instance...")

            # Create predictor instance
            normalization_method = model_data.normalization_method or "none"
            stain_method = model_data.stain_normalization_method
            norm_image = model_data.normalization_image

            # Handle normalization image path
            target_image_path = None
            if norm_image:
                models_path = Path("models")
                target_image_path = str(
                    models_path / model_data.model_guid / norm_image)
                self.status_signal.emit(
                    f"Using normalization image: {target_image_path}")

            # Debug model structure
            self.status_signal.emit(f"Model type: {type(model).__name__}")
            if hasattr(model, "arc"):
                self.status_signal.emit(
                    f"Model has arc attribute of type: {type(model.arc).__name__}"
                )

            # Create predictor
            predictor = ImageSegmentationPredictor(
                model=model,
                device=self.device,
                img_size=model_data.image_size,
                normalization_method=normalization_method,
                stain_normalization_method=stain_method,
                target_image_path=target_image_path,
                status_signal=self.status_signal,
            )

            self.status_signal.emit("Predictor created successfully")
            return predictor

        except Exception as e:
            self.status_signal.emit(f"Error creating predictor: {str(e)}")
            import traceback

            self.status_signal.emit(f"Traceback: {traceback.format_exc()}")
            return None

    def _process_image(self, image_path, save_path):
        """Process a single image."""
        try:
            # Preprocess image
            image_tensor = self.predictor.preprocess_image(image_path)

            if image_tensor is None:
                self.status_signal.emit(
                    f"Error preprocessing image {image_path}")
                return False

            # Generate prediction
            mask = self.predictor.predict_mask(image_tensor)

            # Check if mask is valid
            if mask is None or mask.size == 0:
                self.status_signal.emit(
                    f"Error: Empty mask generated for {image_path}")
                return False

            # Save mask
            self.predictor.save_mask(mask, save_path)

            return True
        except Exception as e:
            self.status_signal.emit(f"Error processing {image_path}: {str(e)}")
            return False
