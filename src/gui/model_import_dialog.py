import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget)


class ModelImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Model")
        self.resize(400, 600)

        layout = QVBoxLayout(self)

        # File selection
        file_layout = QHBoxLayout()
        self.file_path = QLineEdit()
        self.file_path.setReadOnly(True)
        file_layout.addWidget(self.file_path)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_button)

        layout.addLayout(file_layout)

        # Preview section
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # Model information preview
        preview_layout.addWidget(QLabel("Model Information:"))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        preview_layout.addWidget(self.info_text)

        # Image previews
        images_layout = QHBoxLayout()

        # Example image preview
        example_layout = QVBoxLayout()
        example_layout.addWidget(QLabel("Example Image:"))
        self.example_image = QLabel()
        self.example_image.setFixedSize(200, 200)
        self.example_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        example_layout.addWidget(self.example_image)
        images_layout.addLayout(example_layout)

        # Example mask preview
        mask_layout = QVBoxLayout()
        mask_layout.addWidget(QLabel("Example Mask:"))
        self.example_mask = QLabel()
        self.example_mask.setFixedSize(200, 200)
        self.example_mask.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_layout.addWidget(self.example_mask)
        images_layout.addLayout(mask_layout)

        preview_layout.addLayout(images_layout)
        layout.addWidget(preview_widget)

        # Status message
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        # Buttons
        buttons_layout = QHBoxLayout()
        self.import_button = QPushButton("Import")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self.accept)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(self.import_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Archive", "", "ZIP files (*.zip)"
        )
        if file_path:
            self.file_path.setText(file_path)
            self._validate_and_preview()

    def _validate_and_preview(self):
        import os
        import tempfile
        import zipfile

        self.status_label.setText("Validating archive...")
        self.import_button.setEnabled(False)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract archive
                with zipfile.ZipFile(self.file_path.text(), "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Look for description.json
                description_file = Path(temp_dir) / "description.json"
                if not description_file.exists():
                    raise ValueError("description.json not found in archive")

                # Load and display model information
                with open(description_file) as f:
                    description = json.load(f)

                info_text = f"""
                Model Name: {description.get('model_name', 'N/A')}
                Architecture: {description.get('model_architecture', 'N/A')}
                Encoder: {description.get('model_encoder', 'N/A')}
                Number of Classes: {description.get('model_num_classes', 'N/A')}
                Image Size: {description.get('image_size', 'N/A')}
                Normalization Method: {description.get('normalization_method', 'N/A')}
                Staining: {description.get('staining', 'N/A')}
                Model class names: {description.get('model_class_names', 'N/A')}
                Model version: {description.get('model_version', 'N/A')}
                Model create date: {description.get('create_date', 'N/A')}
                """
                self.info_text.setText(info_text)

                # Load and display example images
                example_image = None
                example_mask = None

                # Get image and mask filenames from description.json
                image_example_name = description.get("training_image_example")
                mask_example_name = description.get(
                    "training_image_mask_example")

                if image_example_name:
                    image_path = Path(temp_dir) / image_example_name
                    if image_path.exists() and image_path.suffix.lower() in [
                        ".tif",
                        ".jpeg",
                        ".png",
                    ]:
                        example_image = image_path

                if mask_example_name:
                    mask_path = Path(temp_dir) / mask_example_name
                    if mask_path.exists() and mask_path.suffix.lower() in [
                        ".tif",
                        ".jpeg",
                        ".png",
                    ]:
                        example_mask = mask_path

                if example_image:
                    pixmap = QPixmap(str(example_image))
                    self.example_image.setPixmap(
                        pixmap.scaled(
                            200,
                            200,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )

                if example_mask:
                    pixmap = QPixmap(str(example_mask))
                    self.example_mask.setPixmap(
                        pixmap.scaled(
                            200,
                            200,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )

                self.status_label.setText("Archive validated successfully")
                self.import_button.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.import_button.setEnabled(False)

    def get_file_path(self):
        """Return the selected file path."""
        return self.file_path.text()
