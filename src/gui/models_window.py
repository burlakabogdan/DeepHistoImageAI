import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QHBoxLayout, QHeaderView, QMainWindow,
                             QMessageBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from src.core import get_logger
from src.core.models import DeepLearningModel


class ModelsWindow(QMainWindow):
    def __init__(self, parent=None, model_manager=None):
        super().__init__(parent)
        self.setWindowTitle("Model Management")
        self.resize(800, 600)

        # Store the model manager and parent
        self.model_manager = model_manager
        self.parent_window = parent

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create toolbar
        toolbar = QHBoxLayout()

        # Add model button
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_model)
        toolbar.addWidget(add_button)

        # Edit model button
        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self._edit_model)
        self.edit_button.setEnabled(False)
        toolbar.addWidget(self.edit_button)

        # Delete model button
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_model)
        self.delete_button.setEnabled(False)
        toolbar.addWidget(self.delete_button)

        # Import model button
        import_button = QPushButton("Get models form server")

        import_button.clicked.connect(self._import_model)
        toolbar.addWidget(import_button)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            [
                "Model Name",
                "Architecture",
                "Encoder",
                "Classes",
                "Description",
                "Created At",
            ]
        )

        # Set table properties
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)

        # Connect selection changed signal
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addWidget(self.table)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load models from the database and populate the table."""
        if not self.model_manager or not self.model_manager.db_session:
            QMessageBox.warning(
                self,
                "Database Error",
                "No database connection available.")
            return

        # Clear the table
        self.table.setRowCount(0)

        try:
            # Query all models from the database
            models = self.model_manager.db_session.query(
                DeepLearningModel).all()

            # Add each model to the table
            for model in models:
                row_position = self.table.rowCount()
                self.table.insertRow(row_position)

                # Set model data in the table
                self.table.setItem(
                    row_position, 0, QTableWidgetItem(
                        model.model_name))
                self.table.setItem(
                    row_position, 1, QTableWidgetItem(
                        model.model_architecture))
                self.table.setItem(
                    row_position, 2, QTableWidgetItem(
                        model.model_encoder))
                self.table.setItem(row_position, 3, QTableWidgetItem(
                    str(model.model_num_classes)))
                self.table.setItem(
                    row_position, 4, QTableWidgetItem(
                        model.model_description))

                # Format the creation date
                create_date = (model.create_date.strftime(
                    "%Y-%m-%d %H:%M:%S") if model.create_date else "")
                self.table.setItem(
                    row_position, 5, QTableWidgetItem(create_date))

        except Exception as e:
            QMessageBox.critical(
                self,
                "Database Error",
                f"Failed to load models from database:\n{str(e)}",
            )

    def _on_selection_changed(self):
        # Enable/disable buttons based on selection
        has_selection = len(self.table.selectedItems()) > 0
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)

    def _add_model(self):
        """Open the add model dialog."""
        from src.gui.model_edit_dialog import ModelEditDialog

        dialog = ModelEditDialog(self, None)
        if dialog.exec():
            # Get the model data from the dialog
            model_data = dialog.get_model_data()

            try:
                # Create a new model
                new_model = DeepLearningModel(
                    model_guid=model_data["model_guid"],
                    model_name=model_data["model_name"],
                    model_file_name=model_data["model_file_name"],
                    model_description=model_data["model_description"],
                    model_architecture=model_data["model_architecture"],
                    model_encoder=model_data["model_encoder"],
                    model_weights=model_data["model_weights"],
                    model_in_channels=model_data["model_in_channels"],
                    model_num_classes=model_data["model_num_classes"],
                    model_class_names=model_data["model_class_names"],
                    model_class_colours=model_data["model_class_colours"],
                    staining=model_data["staining"],
                    image_size=model_data["image_size"],
                    normalization_method=model_data["normalization_method"],
                    stain_normalization_method=model_data["stain_normalization_method"],
                    training_image_example=model_data["training_image_example"],
                    training_image_mask_example=model_data["training_image_mask_example"],
                    normalization_image=model_data["normalization_image"],
                    model_version=model_data["model_version"],
                )

                # Add to database
                self.model_manager.db_session.add(new_model)
                self.model_manager.db_session.commit()

                # Create model directory in models folder
                model_dir = Path(
                    self.model_manager.models_path) / model_data["model_guid"]
                model_dir.mkdir(parents=True, exist_ok=True)

                # Copy model file if it exists and is a file path (not just a
                # filename)
                model_file_path = model_data["model_file_name"]
                print(f"Model file path: {model_file_path}")

                # Check if this is a full path to an existing file
                if os.path.isfile(model_file_path):
                    # Copy the model file to the model directory
                    target_file = model_dir / os.path.basename(model_file_path)
                    print(f"Copying model file to: {target_file}")
                    shutil.copy2(model_file_path, target_file)
                    # Update the model_file_name in the database to be just the
                    # filename
                    new_model.model_file_name = os.path.basename(
                        model_file_path)
                    self.model_manager.db_session.commit()
                else:
                    print(
                        f"Model file path is not a valid file: {model_file_path}")

                # Copy training image example if it exists
                if model_data["training_image_example"] and os.path.isfile(
                    model_data["training_image_example"]
                ):
                    target_file = model_dir / \
                        os.path.basename(model_data["training_image_example"])
                    shutil.copy2(
                        model_data["training_image_example"],
                        target_file)
                    new_model.training_image_example = os.path.basename(
                        model_data["training_image_example"]
                    )

                # Copy training image mask example if it exists
                if model_data["training_image_mask_example"] and os.path.isfile(
                        model_data["training_image_mask_example"]):
                    target_file = model_dir / os.path.basename(
                        model_data["training_image_mask_example"]
                    )
                    shutil.copy2(
                        model_data["training_image_mask_example"],
                        target_file)
                    new_model.training_image_mask_example = os.path.basename(
                        model_data["training_image_mask_example"]
                    )

                # Copy normalization image if it exists
                if model_data["normalization_image"] and os.path.isfile(
                    model_data["normalization_image"]
                ):
                    target_file = model_dir / \
                        os.path.basename(model_data["normalization_image"])
                    shutil.copy2(
                        model_data["normalization_image"], target_file)
                    new_model.normalization_image = os.path.basename(
                        model_data["normalization_image"]
                    )

                # Commit the updated file paths
                self.model_manager.db_session.commit()

                # Reload the models
                self._load_models()

                # Notify parent window to reload models
                if hasattr(self.parent_window, "_load_models"):
                    self.parent_window._load_models()

            except Exception as e:
                self.model_manager.db_session.rollback()
                QMessageBox.critical(
                    self, "Database Error", f"Failed to add model: {str(e)}")

    def _edit_model(self):
        """Open the edit model dialog."""
        selected_rows = self.table.selectedItems()
        if selected_rows:
            # Get the model name from the first column
            model_name = self.table.item(selected_rows[0].row(), 0).text()

            try:
                # Get the model from the database
                model = (
                    self.model_manager.db_session.query(DeepLearningModel)
                    .filter_by(model_name=model_name)
                    .first()
                )

                if model:
                    import os
                    import shutil
                    from pathlib import Path

                    from src.gui.model_edit_dialog import ModelEditDialog

                    # Create a dictionary with model data
                    model_data = {
                        "model_guid": model.model_guid,
                        "model_name": model.model_name,
                        "model_file_name": model.model_file_name,
                        "model_description": model.model_description or "",
                        "model_architecture": model.model_architecture,
                        "model_encoder": model.model_encoder,
                        "model_weights": model.model_weights or "",
                        "model_in_channels": model.model_in_channels,
                        "model_num_classes": model.model_num_classes,
                        "model_class_names": model.model_class_names or [],
                        "model_class_colours": model.model_class_colours or [],
                        "staining": model.staining or "",
                        "image_size": model.image_size,
                        "normalization_method": model.normalization_method,
                        "stain_normalization_method": model.stain_normalization_method,
                        "training_image_example": model.training_image_example or "",
                        "training_image_mask_example": model.training_image_mask_example or "",
                        "normalization_image": model.normalization_image or "",
                        "model_version": model.model_version or "",
                        "create_date": (
                            model.create_date.isoformat() if model.create_date else ""),
                    }

                    dialog = ModelEditDialog(self, model_data)
                    if dialog.exec():
                        # Get the updated model data
                        updated_model_data = dialog.get_model_data()

                        # Update the model
                        model.model_name = updated_model_data["model_name"]
                        model.model_description = updated_model_data["model_description"]
                        model.model_architecture = updated_model_data["model_architecture"]
                        model.model_encoder = updated_model_data["model_encoder"]
                        model.model_weights = updated_model_data["model_weights"]
                        model.model_in_channels = updated_model_data["model_in_channels"]
                        model.model_num_classes = updated_model_data["model_num_classes"]
                        model.model_class_names = updated_model_data["model_class_names"]
                        model.model_class_colours = updated_model_data["model_class_colours"]
                        model.staining = updated_model_data["staining"]
                        model.image_size = updated_model_data["image_size"]
                        model.normalization_method = updated_model_data["normalization_method"]
                        model.stain_normalization_method = updated_model_data[
                            "stain_normalization_method"
                        ]
                        model.model_version = updated_model_data["model_version"]

                        # Create model directory in models folder if it doesn't
                        # exist
                        model_dir = Path(
                            self.model_manager.models_path) / model.model_guid
                        model_dir.mkdir(parents=True, exist_ok=True)

                        # Handle model file
                        model_file_path = updated_model_data["model_file_name"]
                        print(f"Edit - Model file path: {model_file_path}")
                        print(
                            f"Edit - Is file: {os.path.isfile(model_file_path)}")

                        if model_file_path != model.model_file_name and os.path.isfile(
                                model_file_path):
                            # Copy the model file to the model directory
                            target_file = model_dir / \
                                os.path.basename(model_file_path)
                            print(
                                f"Edit - Copying model file to: {target_file}")
                            shutil.copy2(model_file_path, target_file)
                            model.model_file_name = os.path.basename(
                                model_file_path)
                        else:
                            print(
                                f"Edit - Model file not copied: {model_file_path} vs {model.model_file_name}"
                            )

                        # Handle training image example
                        if updated_model_data[
                            "training_image_example"
                        ] != model.training_image_example and os.path.isfile(
                            updated_model_data["training_image_example"]
                        ):
                            target_file = model_dir / os.path.basename(
                                updated_model_data["training_image_example"]
                            )
                            shutil.copy2(
                                updated_model_data["training_image_example"],
                                target_file,
                            )
                            model.training_image_example = os.path.basename(
                                updated_model_data["training_image_example"]
                            )

                        # Handle training image mask example
                        if updated_model_data[
                            "training_image_mask_example"
                        ] != model.training_image_mask_example and os.path.isfile(
                            updated_model_data["training_image_mask_example"]
                        ):
                            target_file = model_dir / os.path.basename(
                                updated_model_data["training_image_mask_example"]
                            )
                            shutil.copy2(
                                updated_model_data["training_image_mask_example"], target_file, )
                            model.training_image_mask_example = os.path.basename(
                                updated_model_data["training_image_mask_example"]
                            )

                        # Handle normalization image
                        if updated_model_data[
                            "normalization_image"
                        ] != model.normalization_image and os.path.isfile(
                            updated_model_data["normalization_image"]
                        ):
                            target_file = model_dir / os.path.basename(
                                updated_model_data["normalization_image"]
                            )
                            shutil.copy2(
                                updated_model_data["normalization_image"], target_file)
                            model.normalization_image = os.path.basename(
                                updated_model_data["normalization_image"]
                            )

                        # Commit changes
                        self.model_manager.db_session.commit()

                        # Reload the models
                        self._load_models()

                        # Notify parent window to reload models
                        if hasattr(self.parent_window, "_load_models"):
                            self.parent_window._load_models()
                else:
                    QMessageBox.warning(
                        self, "Error", f"Model '{model_name}' not found")

            except Exception as e:
                self.model_manager.db_session.rollback()
                QMessageBox.critical(
                    self, "Database Error", f"Failed to edit model: {str(e)}")

    def _delete_model(self):
        """Delete the selected model."""
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            model_name = self.table.item(selected_row, 0).text()
            reply = QMessageBox.question(
                self,
                "Delete Model",
                f"Are you sure you want to delete model '{model_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    # Get the model from the database
                    model = (
                        self.model_manager.db_session.query(DeepLearningModel)
                        .filter_by(model_name=model_name)
                        .first()
                    )

                    if model:
                        # Store the model GUID before deleting from database
                        model_guid = model.model_guid

                        # Delete the model from database
                        self.model_manager.db_session.delete(model)
                        self.model_manager.db_session.commit()

                        # Ask if the user also wants to delete the model files
                        file_reply = QMessageBox.question(
                            self,
                            "Delete Model Files",
                            f"Do you also want to delete all model files from disk?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No,
                        )

                        if file_reply == QMessageBox.StandardButton.Yes:
                            # Delete the model directory
                            model_dir = Path(
                                self.model_manager.models_path) / model_guid
                            if model_dir.exists() and model_dir.is_dir():
                                try:
                                    shutil.rmtree(model_dir)
                                    QMessageBox.information(
                                        self, "Files Deleted", f"Model files have been deleted from {model_dir}", )
                                except Exception as e:
                                    QMessageBox.warning(
                                        self, "File Deletion Error", f"Failed to delete model files: {str(e)}", )

                        # Reload the models
                        self._load_models()

                        # Notify parent window to reload models if applicable
                        if hasattr(self.parent_window, "_load_models"):
                            self.parent_window._load_models()
                    else:
                        QMessageBox.warning(
                            self,
                            "Model Not Found",
                            f"Could not find model '{model_name}' in the database.",
                        )

                except Exception as e:
                    self.model_manager.db_session.rollback()
                    QMessageBox.critical(
                        self, "Database Error", f"Failed to delete model: {str(e)}")

    def _import_model(self):
        # This will be implemented to open the import model dialog
        pass
