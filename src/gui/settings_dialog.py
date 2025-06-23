import configparser
import os
from pathlib import Path

import torch
from PyQt6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                             QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
                             QWidget)

from src.core import get_logger


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = get_logger(__name__)
        self.setWindowTitle("Settings")
        self.resize(500, 400)

        # Load settings from config file
        self.config = configparser.ConfigParser()
        self.config_file = Path("config/settings.ini")
        if self.config_file.exists():
            self.config.read(self.config_file)

        # Create the main layout
        main_layout = QVBoxLayout(self)

        # Create tabs
        self.tabs = QTabWidget()
        self.paths_tab = QWidget()
        self.default_tab = QWidget()
        self.logging_tab = QWidget()

        self.tabs.addTab(self.paths_tab, "Paths")
        self.tabs.addTab(self.default_tab, "Default")
        self.tabs.addTab(self.logging_tab, "Logging")

        # Set up each tab
        self._setup_paths_tab()
        self._setup_default_tab()
        self._setup_logging_tab()

        # Add tabs to main layout
        main_layout.addWidget(self.tabs)

        # Add buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def _setup_paths_tab(self):
        layout = QFormLayout(self.paths_tab)

        # Input path
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit(
            self._get_config_value("Paths", "input_path", "input_folder")
        )
        input_layout.addWidget(self.input_path_edit)
        browse_input_btn = QPushButton("Browse...")
        browse_input_btn.clicked.connect(
            lambda: self._browse_folder(
                self.input_path_edit,
                "Select Input Folder"))
        input_layout.addWidget(browse_input_btn)
        layout.addRow("Input Path:", input_layout)

        # Output path
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit(
            self._get_config_value("Paths", "output_path", "output_folder")
        )
        output_layout.addWidget(self.output_path_edit)
        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(
            lambda: self._browse_folder(
                self.output_path_edit,
                "Select Output Folder"))
        output_layout.addWidget(browse_output_btn)
        layout.addRow("Output Path:", output_layout)

        # Models path
        models_layout = QHBoxLayout()
        self.models_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "models_path", "models"))
        models_layout.addWidget(self.models_path_edit)
        browse_models_btn = QPushButton("Browse...")
        browse_models_btn.clicked.connect(
            lambda: self._browse_folder(
                self.models_path_edit,
                "Select Models Folder"))
        models_layout.addWidget(browse_models_btn)
        layout.addRow("Models Path:", models_layout)

        # Config path
        config_layout = QHBoxLayout()
        self.config_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "config_path", "config"))
        config_layout.addWidget(self.config_path_edit)
        browse_config_btn = QPushButton("Browse...")
        browse_config_btn.clicked.connect(
            lambda: self._browse_folder(
                self.config_path_edit,
                "Select Config Folder"))
        config_layout.addWidget(browse_config_btn)
        layout.addRow("Config Path:", config_layout)

        # Log path
        log_layout = QHBoxLayout()
        self.log_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "log_path", "logs"))
        log_layout.addWidget(self.log_path_edit)
        browse_log_btn = QPushButton("Browse...")
        browse_log_btn.clicked.connect(
            lambda: self._browse_folder(
                self.log_path_edit,
                "Select Log Folder"))
        log_layout.addWidget(browse_log_btn)
        layout.addRow("Log Path:", log_layout)

        # Data path
        data_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "data_path", "data"))
        data_layout.addWidget(self.data_path_edit)
        browse_data_btn = QPushButton("Browse...")
        browse_data_btn.clicked.connect(
            lambda: self._browse_folder(
                self.data_path_edit,
                "Select Data Folder"))
        data_layout.addWidget(browse_data_btn)
        layout.addRow("Data Path:", data_layout)

        # Docs path
        docs_layout = QHBoxLayout()
        self.docs_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "docs_path", "docs"))
        docs_layout.addWidget(self.docs_path_edit)
        browse_docs_btn = QPushButton("Browse...")
        browse_docs_btn.clicked.connect(
            lambda: self._browse_folder(
                self.docs_path_edit,
                "Select Docs Folder"))
        docs_layout.addWidget(browse_docs_btn)
        layout.addRow("Docs Path:", docs_layout)

        # DB Migrations path
        migrations_layout = QHBoxLayout()
        self.migrations_path_edit = QLineEdit(
            self._get_config_value(
                "Paths",
                "db_migrations_path",
                "src/migrations"))
        migrations_layout.addWidget(self.migrations_path_edit)
        browse_migrations_btn = QPushButton("Browse...")
        browse_migrations_btn.clicked.connect(
            lambda: self._browse_folder(
                self.migrations_path_edit,
                "Select DB Migrations Folder"))
        migrations_layout.addWidget(browse_migrations_btn)
        layout.addRow("DB Migrations Path:", migrations_layout)

        # Tests path
        tests_layout = QHBoxLayout()
        self.tests_path_edit = QLineEdit(
            self._get_config_value(
                "Paths", "tests_path", "src/tests"))
        tests_layout.addWidget(self.tests_path_edit)
        browse_tests_btn = QPushButton("Browse...")
        browse_tests_btn.clicked.connect(
            lambda: self._browse_folder(
                self.tests_path_edit,
                "Select Tests Folder"))
        tests_layout.addWidget(browse_tests_btn)
        layout.addRow("Tests Path:", tests_layout)

    def _setup_default_tab(self):
        layout = QFormLayout(self.default_tab)

        # Device selection
        self.device_combo = QComboBox()
        devices = ["cpu"]

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({gpu_name})")
        self.device_combo.addItems(devices)
        current_device = self._get_config_value("Default", "device", "cpu")
        current_index = 0
        for i, device in enumerate(devices):
            if device.startswith(current_device):
                current_index = i
                break
        self.device_combo.setCurrentIndex(current_index)
        layout.addRow("Device:", self.device_combo)

        # Add GPU warning label if no GPU is found
        if not torch.cuda.is_available():
            gpu_warning = QLabel("GPU not found")
            gpu_warning.setStyleSheet("color: red;")
            layout.addRow(gpu_warning)

        # Number of workers
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, 16)
        self.num_workers_spin.setValue(
            int(self._get_config_value("Default", "num_workers", "2")))
        layout.addRow("Number of Workers:", self.num_workers_spin)

        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(
            int(self._get_config_value("Default", "batch_size", "4")))
        layout.addRow("Batch Size:", self.batch_size_spin)

    def _setup_logging_tab(self):
        layout = QFormLayout(self.logging_tab)

        # Log level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["INFO", "DEBUG"])
        current_log_level = self._get_config_value(
            "Logging", "log_level", "INFO")
        self.log_level_combo.setCurrentText(current_log_level)
        layout.addRow("Log Level:", self.log_level_combo)

        # Log file
        self.log_file_edit = QLineEdit(
            self._get_config_value(
                "Logging",
                "log_file",
                "deep_image_app_{datetime}.log"))
        layout.addRow("Log File:", self.log_file_edit)

        # Max bytes
        self.max_bytes_spin = QSpinBox()
        self.max_bytes_spin.setRange(1024, 1073741824)  # 1KB to 1GB
        self.max_bytes_spin.setSingleStep(1024)
        self.max_bytes_spin.setValue(
            int(self._get_config_value("Logging", "max_bytes", "5242880")))
        layout.addRow("Max Bytes:", self.max_bytes_spin)

        # Backup count
        self.backup_count_spin = QSpinBox()
        self.backup_count_spin.setRange(0, 100)
        self.backup_count_spin.setValue(
            int(self._get_config_value("Logging", "backup_count", "3")))
        layout.addRow("Backup Count:", self.backup_count_spin)

    def _get_config_value(self, section, option, default):
        """Get a value from the config file, or return the default if not found."""
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def _browse_folder(self, line_edit, caption):
        """Open a folder browser dialog and set the selected path in the line edit."""
        folder = QFileDialog.getExistingDirectory(
            self, caption, line_edit.text() or os.path.expanduser("~")
        )
        if folder:
            line_edit.setText(folder)

    def save_settings(self):
        """Save the settings to the config file."""
        # Ensure all sections exist
        for section in ["Paths", "Default", "Logging"]:
            if not self.config.has_section(section):
                self.config.add_section(section)

        # Save Paths settings
        self.config.set("Paths", "input_path", self.input_path_edit.text())
        self.config.set("Paths", "output_path", self.output_path_edit.text())
        self.config.set("Paths", "models_path", self.models_path_edit.text())
        self.config.set("Paths", "config_path", self.config_path_edit.text())
        self.config.set("Paths", "log_path", self.log_path_edit.text())
        self.config.set("Paths", "data_path", self.data_path_edit.text())
        self.config.set("Paths", "docs_path", self.docs_path_edit.text())
        self.config.set(
            "Paths",
            "db_migrations_path",
            self.migrations_path_edit.text())
        self.config.set("Paths", "tests_path", self.tests_path_edit.text())

        # Save Default settings
        # self.config.set("Default", "device", self.device_combo.currentText())
        device_text = self.device_combo.currentText().split()[0]
        self.config.set("Default", "device", device_text)   

        self.config.set(
            "Default", "num_workers", str(
                self.num_workers_spin.value()))
        self.config.set(
            "Default", "batch_size", str(
                self.batch_size_spin.value()))

        # Save Logging settings
        self.config.set(
            "Logging",
            "log_level",
            self.log_level_combo.currentText())
        self.config.set("Logging", "log_file", self.log_file_edit.text())
        self.config.set(
            "Logging", "max_bytes", str(
                self.max_bytes_spin.value()))
        self.config.set(
            "Logging", "backup_count", str(
                self.backup_count_spin.value()))

        # Ensure the config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        # Write the config file
        with open(self.config_file, "w") as f:
            self.config.write(f)

        return True

    def accept(self):
        """Override the accept method to save settings before closing."""
        self.save_settings()
        super().accept()
