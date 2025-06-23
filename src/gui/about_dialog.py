from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class AboutDialog(QDialog):
    """Dialog showing information about the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.resize(500, 300)
        self.setModal(True)

        # Create the main layout
        main_layout = QVBoxLayout(self)

        # Application title
        title_label = QLabel("Deep Histo Image AI")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Application subtitle
        subtitle_label = QLabel(
            "Platform for Artificial Intelligence-based Digital Pathology"
        )
        subtitle_font = QFont()
        subtitle_font.setPointSize(12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle_label)

        # Add some spacing
        main_layout.addSpacing(20)

        # Description
        description_label = QLabel(
            """
            Deep Histo Image is an advanced platform designed to simplify the use of modern artificial intelligence technologies in digital pathology. It enables pathologists, researchers, and developers to efficiently analyze histological images using state-of-the-art deep learning methods.

            Key Features:
            🔬 Segmentation and Classification – Automated tissue structure detection and identification of pathological changes in histological images.
            ⚡ Ease of Use – An intuitive interface allows users to upload, process, and analyze images without requiring deep knowledge of programming or machine learning.
            🧠 Advanced AI Models – Integration of powerful deep learning algorithms (U-Net, DeepLabV3, ...) for accurate analysis.
            📊 Image Normalization and Processing – Automated color normalization methods (Macenko, Reinhard) and preprocessing ensure data consistency.
            🛠 Flexible Integration – API customization to connect with LIS (Laboratory Information Systems) and other platforms.
            🔍 Digital Validation – Support for annotations and expert interaction for result validation.
            ☁ Cloud Storage and Computing – Support for large-scale data processing on local or cloud servers.

            Platform Objectives:
            ✔ Automate routine histological image analysis processes
            ✔ Improve diagnostic accuracy using AI
            ✔ Make modern AI algorithms accessible to doctors and researchers
            ✔ Promote the development of digital pathology and personalized medicine
            """
        )
        description_label.setWordWrap(True)
        description_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        main_layout.addWidget(description_label)

        # Add some spacing
        main_layout.addSpacing(20)

        # Get the application instance and retrieve the version
        app = QApplication.instance()
        version = app.applicationVersion()

        # Version information
        version_label = QLabel(f"Version: {version}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(version_label)

        # Copyright information
        copyright_label = QLabel("© 2025 Deep Histo Image AI Team")
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(copyright_label)

        # Add stretching space
        main_layout.addStretch()

        # Add OK button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        main_layout.addWidget(button_box)
