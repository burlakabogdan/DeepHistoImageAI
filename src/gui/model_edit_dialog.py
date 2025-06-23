import os
import uuid
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtWidgets import *


class ModelEditDialog(QDialog):
    def __init__(self, parent=None, model_data=None):
        super().__init__(parent)
        self.setWindowTitle("Add/Edit Model")
        self.resize(800, 800)
        self.model_data = model_data.copy() if model_data else {}

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create a scroll area for the form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Create a widget to hold the form
        form_widget = QWidget()
        layout = QVBoxLayout(form_widget)

        # Form layout for model properties
        form_layout = QFormLayout()

        # Model Name
        self.name_edit = QLineEdit(self.model_data.get("model_name", ""))
        form_layout.addRow("Model Name:", self.name_edit)

        # Description
        self.description_edit = QTextEdit()
        self.description_edit.setText(
            self.model_data.get(
                "model_description", ""))
        form_layout.addRow("Description:", self.description_edit)

        # Model GUID (read-only if editing)
        self.guid_edit = QLineEdit(
            self.model_data.get(
                "model_guid", str(
                    uuid.uuid4())))
        self.guid_edit.setReadOnly(True)
        form_layout.addRow("Model GUID:", self.guid_edit)

        # Model File Name
        self.file_name_edit = QLineEdit(
            self.model_data.get(
                "model_file_name", ""))
        if not self.file_name_edit.text() and self.name_edit.text():
            default_name = f"{self.name_edit.text().lower().replace(' ', '_')}.pth"
            self.file_name_edit.setText(default_name)
        file_name_layout = QHBoxLayout()
        file_name_layout.addWidget(self.file_name_edit)
        file_name_browse = QPushButton("Browse...")
        file_name_browse.clicked.connect(self._browse_model_file)
        file_name_layout.addWidget(file_name_browse)
        form_layout.addRow("Model File Name:", file_name_layout)

        # Model Version
        self.version_edit = QLineEdit(
            self.model_data.get(
                "model_version", "1.0.0"))
        form_layout.addRow("Model Version:", self.version_edit)

        # Create Date
        self.create_date_edit = QDateTimeEdit()
        if "create_date" in self.model_data and self.model_data["create_date"]:
            self.create_date_edit.setDateTime(
                QDateTime.fromString(
                    self.model_data["create_date"],
                    Qt.DateFormat.ISODate))
        else:
            self.create_date_edit.setDateTime(QDateTime.currentDateTime())
        self.create_date_edit.setReadOnly(True)
        form_layout.addRow("Create Date:", self.create_date_edit)

        # Architecture
        self.architecture_combo = QComboBox()
        self.architecture_combo.addItems(
            ["Unet", "DeepLabV3", "DeepLabV3Plus", "FPN", "PSPNet"])
        if "model_architecture" in self.model_data:
            self.architecture_combo.setCurrentText(
                self.model_data["model_architecture"])
        form_layout.addRow("Architecture:", self.architecture_combo)

        # Encoder
        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(
            [
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "efficientnet-b0",
                "efficientnet-b1",
                "efficientnet-b2",
            ]
        )
        if "model_encoder" in self.model_data:
            self.encoder_combo.setCurrentText(self.model_data["model_encoder"])
        form_layout.addRow("Encoder:", self.encoder_combo)

        # Weights
        self.weights_combo = QComboBox()
        self.weights_combo.addItems(["imagenet", "ssl", "swsl", "none"])
        if "model_weights" in self.model_data:
            self.weights_combo.setCurrentText(self.model_data["model_weights"])
        form_layout.addRow("Encoder Weights:", self.weights_combo)

        # Input Channels
        self.channels_spin = QSpinBox()
        self.channels_spin.setRange(1, 4)
        self.channels_spin.setValue(
            self.model_data.get(
                "model_in_channels", 3))
        form_layout.addRow("Input Channels:", self.channels_spin)

        # Number of Classes
        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(1, 100)
        self.classes_spin.setValue(self.model_data.get("model_num_classes", 1))
        form_layout.addRow("Number of Classes:", self.classes_spin)

        # Class Names
        self.class_names_edit = QLineEdit(
            ",".join(
                self.model_data.get(
                    "model_class_names",
                    [])))
        form_layout.addRow(
            "Class Names (comma-separated):",
            self.class_names_edit)

        # Class Colors
        self.class_colors_edit = QLineEdit(
            ",".join(
                self.model_data.get(
                    "model_class_colours",
                    [])))
        form_layout.addRow(
            "Class Colors (comma-separated hex):",
            self.class_colors_edit)

        # Staining
        self.staining_combo = QComboBox()
        self.staining_combo.addItems(["None", "H-DAB", "H&E", "IHC"])
        if "staining" in self.model_data:
            self.staining_combo.setCurrentText(self.model_data["staining"])
        form_layout.addRow("Staining:", self.staining_combo)

        # Image Size
        self.image_size_spin = QSpinBox()
        self.image_size_spin.setRange(32, 1024)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(self.model_data.get("image_size", 512))
        form_layout.addRow("Image Size:", self.image_size_spin)

        # Normalization Method
        self.norm_method_combo = QComboBox()
        self.norm_method_combo.addItems(
            ["none", "imagenet", "custom", "stain"])
        if "normalization_method" in self.model_data:
            self.norm_method_combo.setCurrentText(
                self.model_data["normalization_method"])
        self.norm_method_combo.currentTextChanged.connect(
            self._on_norm_method_changed)
        form_layout.addRow("Normalization Method:", self.norm_method_combo)

        # Normalization Image
        self.norm_image_edit = QLineEdit(
            self.model_data.get(
                "normalization_image", ""))
        self.norm_image_layout = QHBoxLayout()
        self.norm_image_layout.addWidget(self.norm_image_edit)
        self.norm_image_browse = QPushButton("Browse...")
        self.norm_image_browse.clicked.connect(
            lambda: self._browse_file(
                self.norm_image_edit,
                "Select Normalization Image"))
        self.norm_image_layout.addWidget(self.norm_image_browse)
        form_layout.addRow("Normalization Image:", self.norm_image_layout)

        # self.norm_image_preview = QLabel()
        # self.norm_image_preview.setFixedSize(100, 100)
        # self.norm_image_preview_layout = QHBoxLayout()
        # self.norm_image_preview_layout.addWidget(self.norm_image_preview)

        # self.norm_image_preview.setAlignment(Qt.AlignCenter)
        # self.norm_image_layout.addWidget(self.norm_image_preview)
        # self.norm_image_label = QLabel("Normalization Image:")
        # form_layout.addRow(self.norm_image_label, self.norm_image_layout)

        # Update preview image when text changes
        # self.norm_image_edit.textChanged.connect(self._update_norm_image_preview)

        # Set initial state of normalization image based on normalization method
        # self._update_norm_image_state(self.norm_method_combo.currentText())

        # Stain Normalization Method
        self.stain_norm_combo = QComboBox()
        self.stain_norm_combo.addItems(
            ["none", "macenko", "reinhard", "reinhard_modified"])
        if "stain_normalization_method" in self.model_data:
            self.stain_norm_combo.setCurrentText(
                self.model_data["stain_normalization_method"])
        else:
            self.stain_norm_combo.setCurrentText("none")
        self.stain_norm_combo.setEnabled(
            self.norm_method_combo.currentText() == "stain")
        form_layout.addRow(
            "Stain Normalization Method:",
            self.stain_norm_combo)

        # Image paths group
        image_paths_group = QGroupBox("Example Images")
        image_paths_layout = QFormLayout(image_paths_group)

        # Training Image Example
        self.training_image_edit = QLineEdit(
            self.model_data.get("training_image_example", ""))
        training_image_layout = QHBoxLayout()
        training_image_layout.addWidget(self.training_image_edit)
        training_image_browse = QPushButton("Browse...")
        training_image_browse.clicked.connect(
            lambda: self._browse_file(
                self.training_image_edit,
                "Select Training Image Example"))
        training_image_layout.addWidget(training_image_browse)
        image_paths_layout.addRow(
            "Training Image Example:",
            training_image_layout)

        # Training Image Mask Example
        self.training_mask_edit = QLineEdit(
            self.model_data.get(
                "training_image_mask_example", ""))
        training_mask_layout = QHBoxLayout()
        training_mask_layout.addWidget(self.training_mask_edit)
        training_mask_browse = QPushButton("Browse...")
        training_mask_browse.clicked.connect(
            lambda: self._browse_file(
                self.training_mask_edit,
                "Select Training Image Mask Example"))
        training_mask_layout.addWidget(training_mask_browse)
        image_paths_layout.addRow(
            "Training Image Mask Example:",
            training_mask_layout)

        # Add form layouts to main layout
        layout.addLayout(form_layout)
        layout.addWidget(image_paths_group)

        # Set the form widget as the scroll area's widget
        scroll_area.setWidget(form_widget)
        main_layout.addWidget(scroll_area)

        # Buttons
        buttons_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._validate_and_accept)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons_layout.addWidget(save_button)
        buttons_layout.addWidget(cancel_button)
        main_layout.addLayout(buttons_layout)

    def _browse_file(self, line_edit, caption):
        """Open file browser dialog and set the selected path to the line edit."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption,
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)",
        )
        if file_path:
            line_edit.setText(file_path)

    def _browse_model_file(self):
        """Open file browser dialog for model files and set the selected path to the line edit."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.pt *.pth);;All Files (*)")
        if file_path:
            # Store the full path for the model file
            self.file_name_edit.setText(file_path)

    def _validate_and_accept(self):
        """Validate form data before accepting."""
        # Basic validation
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Validation Error",
                                "Model name is required")
            return

        if not self.file_name_edit.text().strip():
            QMessageBox.warning(
                self,
                "Validation Error",
                "Model file name is required")
            return

        class_names = [name.strip() for name in self.class_names_edit.text().split(
            ",") if name.strip()]
        if len(class_names) != self.classes_spin.value():
            QMessageBox.warning(
                self,
                "Validation Error",
                f"Number of class names ({len(class_names)}) must match number of classes ({self.classes_spin.value()})",
            )
            return

        self.accept()

    def _on_norm_method_changed(self, text):
        """Enable or disable stain normalization method based on normalization method."""
        self.stain_norm_combo.setEnabled(text == "stain")
        if text != "stain":
            self.stain_norm_combo.setCurrentText("none")

        # Update normalization image state
        self._update_norm_image_state(text)

    def _update_norm_image_state(self, norm_method):
        """
        Enable or disable normalization image based on normalization method.
        The normalization image field is only enabled when the normalization method is 'stain'.

        Args:
            norm_method: The current normalization method
        """
        is_enabled = norm_method == "stain"
        self.norm_image_edit.setEnabled(is_enabled)
        self.norm_image_browse.setEnabled(is_enabled)
        self.norm_image_label.setEnabled(is_enabled)

    def get_model_data(self):
        """Get the form data as a dictionary."""
        class_names = [name.strip() for name in self.class_names_edit.text().split(
            ",") if name.strip()]
        class_colors = [color.strip() for color in self.class_colors_edit.text().split(
            ",") if color.strip()]

        # Convert 'none' to None for model_weights
        weights = self.weights_combo.currentText()
        if weights.lower() == "none":
            weights = None

        # Convert 'None' to None for staining
        staining = self.staining_combo.currentText()
        if staining == "None":
            staining = None

        return {
            "model_guid": self.guid_edit.text(),
            "model_name": self.name_edit.text().strip(),
            "model_file_name": self.file_name_edit.text().strip(),
            "model_description": self.description_edit.toPlainText().strip(),
            "model_architecture": self.architecture_combo.currentText(),
            "model_encoder": self.encoder_combo.currentText(),
            "model_weights": weights,
            "model_in_channels": self.channels_spin.value(),
            "model_num_classes": self.classes_spin.value(),
            "model_class_names": class_names,
            "model_class_colours": class_colors,
            "staining": staining,
            "image_size": self.image_size_spin.value(),
            "normalization_method": self.norm_method_combo.currentText(),
            "stain_normalization_method": self.stain_norm_combo.currentText(),
            "training_image_example": self.training_image_edit.text(),
            "training_image_mask_example": self.training_mask_edit.text(),
            "normalization_image": self.norm_image_edit.text(),
            "model_version": self.version_edit.text(),
            "create_date": self.create_date_edit.dateTime().toString(
                Qt.DateFormat.ISODate),
        }
