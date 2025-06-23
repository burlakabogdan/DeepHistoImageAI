import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QScrollArea, QSizePolicy,
                             QVBoxLayout, QWidget)

from src.core import ModelManager, get_logger


class ImageMaskViewer(QDialog):
    """Dialog for viewing input images and their corresponding masks."""

    def __init__(
            self,
            parent,
            input_path: str,
            output_path: str,
            model_manager: ModelManager):
        super().__init__(parent)
        self.setWindowTitle("Image and Mask Viewer")
        self.resize(1000, 700)  # Initial size, but window will be resizable

        # Initialize paths and model manager
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.model_manager = model_manager
        self.logger = get_logger(__name__)

        # Initialize image-mask pairs
        self.image_mask_pairs = self._get_image_mask_pairs()
        self.current_index = 0

        # Create UI
        self._create_ui()

        # Load initial image-mask pair if available
        if self.image_mask_pairs:
            self._load_current_pair()
        else:
            self._show_no_images_message()

    def _create_ui(self):
        """Create the user interface."""
        main_layout = QVBoxLayout(self)

        # Title labels
        title_layout = QHBoxLayout()
        image_title = QLabel("Original Image")
        image_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        mask_title = QLabel("Predicted Mask")
        mask_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mask_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        title_layout.addWidget(image_title)
        title_layout.addWidget(mask_title)

        # Image and mask display area
        display_layout = QHBoxLayout()

        # Image display
        image_container = QWidget()
        image_layout = QVBoxLayout(image_container)

        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        self.image_scroll_area.setWidget(self.image_label)

        self.image_info_label = QLabel("No image selected")
        self.image_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_layout.addWidget(self.image_scroll_area)
        image_layout.addWidget(self.image_info_label)

        # Mask display
        mask_container = QWidget()
        mask_layout = QVBoxLayout(mask_container)

        self.mask_scroll_area = QScrollArea()
        self.mask_scroll_area.setWidgetResizable(True)
        self.mask_label = QLabel("No mask")
        self.mask_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding)
        self.mask_scroll_area.setWidget(self.mask_label)

        self.mask_info_label = QLabel("No mask selected")
        self.mask_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        mask_layout.addWidget(self.mask_scroll_area)
        mask_layout.addWidget(self.mask_info_label)

        display_layout.addWidget(image_container)
        display_layout.addWidget(mask_container)

        # Navigation buttons
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self._show_previous)
        self.prev_button.setMinimumWidth(100)

        self.image_counter_label = QLabel("0/0")
        self.image_counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_counter_label.setMinimumWidth(80)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._show_next)
        self.next_button.setMinimumWidth(100)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setMinimumWidth(100)

        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.image_counter_label)
        button_layout.addWidget(self.next_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)

        # Add layouts to main layout
        main_layout.addLayout(title_layout)
        main_layout.addLayout(display_layout, 1)
        main_layout.addLayout(button_layout)

        # Update UI state
        self._update_ui_state()

    def _get_image_mask_pairs(self) -> List[Tuple[Path, Optional[Path]]]:
        """Get pairs of input images and corresponding masks."""
        pairs = []

        # Get all image files from input directory
        input_files = []
        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            input_files.extend(list(self.input_path.glob(f"*{ext}")))

        # Sort input files by name
        input_files.sort()

        # For each input file, find corresponding mask file
        for input_file in input_files:
            # Mask file has same name but .tif extension in output directory
            mask_file = self.output_path / f"{input_file.stem}.tif"

            # Add pair to list (mask might not exist yet)
            pairs.append(
                (input_file, mask_file if mask_file.exists() else None))

        return pairs

    def _load_current_pair(self):
        """Load and display the current image-mask pair."""
        if not self.image_mask_pairs or self.current_index >= len(
                self.image_mask_pairs):
            return

        # Get current pair
        input_file, mask_file = self.image_mask_pairs[self.current_index]

        # Load and display input image
        self._load_image(input_file, self.image_label, is_mask=False)
        self.image_info_label.setText(f"Image: {input_file.name}")

        # Load and display mask if it exists
        if mask_file and mask_file.exists():
            self._load_image(mask_file, self.mask_label, is_mask=True)
            self.mask_info_label.setText(f"Mask: {mask_file.name}")
        else:
            self.mask_label.setText("No mask available")
            self.mask_label.setPixmap(QPixmap())
            self.mask_info_label.setText("No mask available")

        # Update counter
        self.image_counter_label.setText(
            f"{self.current_index + 1}/{len(self.image_mask_pairs)}")

        # Update UI state
        self._update_ui_state()

    def _load_image(self, image_path: Path, label: QLabel, is_mask=False):
        """Load an image from path and display it in the given label."""
        try:
            if is_mask:
                # For masks, we need to apply color mapping
                pixmap = self._colorize_mask(image_path)
                if pixmap is None:
                    label.setText(f"Error loading mask: {image_path.name}")
                    return
            else:
                # For regular images, load directly
                image = QImage(str(image_path))
                if image.isNull():
                    label.setText(f"Error loading image: {image_path.name}")
                    return
                pixmap = QPixmap.fromImage(image)

            # Set pixmap to label
            label.setPixmap(pixmap)
            label.setScaledContents(False)

            # Scale the pixmap to fit the scroll area while maintaining aspect
            # ratio
            self._scale_image_to_fit(label)

            # Log
            self.logger.info(
                f"Loaded {'mask' if is_mask else 'image'}: {image_path}")

        except Exception as e:
            self.logger.error(
                f"Error loading {'mask' if is_mask else 'image'} {image_path}: {str(e)}"
            )
            label.setText(
                f"Error loading {'mask' if is_mask else 'image'}: {str(e)}")

    def _colorize_mask(self, mask_path: Path) -> Optional[QPixmap]:
        """Apply color mapping to a mask image."""
        try:
            import numpy as np
            from PIL import Image

            # Define default color mapping for mask classes
            # Format: {class_index: (R, G, B)}
            default_color_map = {
                0: (0, 0, 0),  # Background - Black
                1: (255, 0, 0),  # Class 1 - Red
                2: (0, 255, 0),  # Class 2 - Green
                3: (0, 0, 255),  # Class 3 - Blue
                4: (255, 255, 0),  # Class 4 - Yellow
                5: (255, 0, 255),  # Class 5 - Magenta
                6: (0, 255, 255),  # Class 6 - Cyan
                7: (128, 0, 0),  # Class 7 - Dark Red
                8: (0, 128, 0),  # Class 8 - Dark Green
                9: (0, 0, 128),  # Class 9 - Dark Blue
            }

            color_map = default_color_map.copy()
            class_names = {}

            # Try to get color mapping from the model
            if hasattr(
                    self,
                    "model_manager") and self.model_manager.current_model:
                model_data = self.model_manager.current_model.get("metadata")
                if model_data:
                    # Get class colors from model_class_colours (stored as hex)
                    if model_data.model_class_colours:
                        try:
                            custom_colors = {}
                            for i, hex_color in enumerate(
                                    model_data.model_class_colours):
                                if hex_color:
                                    # Convert hex to RGB
                                    hex_color = hex_color.lstrip("#")
                                    r = int(hex_color[0:2], 16)
                                    g = int(hex_color[2:4], 16)
                                    b = int(hex_color[4:6], 16)
                                    custom_colors[i] = (r, g, b)

                            if custom_colors:
                                # Update color map with custom colors
                                color_map.update(custom_colors)
                                self.logger.info(
                                    f"Using color mapping from model: {color_map}")
                        except Exception as e:
                            self.logger.warning(
                                f"Could not parse color mapping from model: {e}")

                    # Get class names if available
                    if model_data.model_class_names:
                        try:
                            for i, name in enumerate(
                                    model_data.model_class_names):
                                if name:
                                    class_names[i] = name
                        except Exception as e:
                            self.logger.warning(
                                f"Could not parse class names from model: {e}")

            # Load mask as numpy array
            mask_img = Image.open(str(mask_path))
            mask = np.array(mask_img)

            # Handle different mask formats
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                # If mask is already RGB, return it directly
                self.logger.info(
                    f"Mask is already RGB, returning directly: {mask_path}")
                return QPixmap(str(mask_path))

            # Create RGB image with the same shape as mask but with 3 channels
            if len(mask.shape) == 2:
                height, width = mask.shape
            else:
                height, width, _ = mask.shape
                mask = mask[:, :, 0]  # Take first channel if multi-channel

            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

            # Apply color mapping
            unique_values = np.unique(mask)
            self.logger.info(f"Mask unique values: {unique_values}")

            for class_idx in unique_values:
                if class_idx in color_map:
                    colored_mask[mask == class_idx] = color_map[class_idx]
                else:
                    # Use a default color for unknown classes (red)
                    colored_mask[mask == class_idx] = (255, 0, 0)
                    self.logger.warning(
                        f"Unknown class index in mask: {class_idx}")

            # Convert numpy array to QImage
            h, w, c = colored_mask.shape
            bytes_per_line = 3 * w
            q_img = QImage(
                colored_mask.data,
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888)

            # Create pixmap from QImage
            pixmap = QPixmap.fromImage(q_img)

            # Update the mask info label with class information
            if len(unique_values) > 0 and unique_values[0] != 0:
                class_info = []
                for class_idx in unique_values:
                    if class_idx == 0:  # Skip background class
                        continue

                    # Get class name if available, otherwise use class index
                    class_name = class_names.get(
                        class_idx, f"Class {class_idx}")

                    # Get color for this class
                    color = color_map.get(class_idx, (255, 0, 0))
                    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

                    class_info.append(
                        f"{class_name}: <span style='color:{color_hex};'>■</span>")

                if class_info:
                    self.mask_info_label.setText(
                        f"Mask: {mask_path.name} - {', '.join(class_info)}"
                    )

            return pixmap

        except Exception as e:
            self.logger.error(f"Error colorizing mask {mask_path}: {str(e)}")
            return None

    def _scale_image_to_fit(self, label: QLabel):
        """Scale the image to fit the scroll area while maintaining aspect ratio."""
        if not label.pixmap() or label.pixmap().isNull():
            return

        # Get the original pixmap
        pixmap = label.pixmap()

        # Get the size of the scroll area viewport
        if label == self.image_label:
            viewport_size = self.image_scroll_area.viewport().size()
        else:
            viewport_size = self.mask_scroll_area.viewport().size()

        # Scale pixmap to fit viewport while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            viewport_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Set the scaled pixmap
        label.setPixmap(scaled_pixmap)

    def _show_previous(self):
        """Show the previous image-mask pair."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_pair()

    def _show_next(self):
        """Show the next image-mask pair."""
        if self.current_index < len(self.image_mask_pairs) - 1:
            self.current_index += 1
            self._load_current_pair()

    def _update_ui_state(self):
        """Update UI state based on current index and available pairs."""
        # Enable/disable navigation buttons
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(
            self.current_index < len(
                self.image_mask_pairs) - 1)

    def _show_no_images_message(self):
        """Show a message when no images are available."""
        self.image_label.setText("No images found in input directory")
        self.mask_label.setText("No masks available")
        self.image_counter_label.setText("0/0")
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def resizeEvent(self, event: QResizeEvent):
        """Handle resize event to update image scaling."""
        super().resizeEvent(event)

        # Rescale images when window is resized
        if self.image_label.pixmap() and not self.image_label.pixmap().isNull():
            self._scale_image_to_fit(self.image_label)

        if self.mask_label.pixmap() and not self.mask_label.pixmap().isNull():
            self._scale_image_to_fit(self.mask_label)
