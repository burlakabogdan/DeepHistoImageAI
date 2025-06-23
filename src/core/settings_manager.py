import configparser
import os
from pathlib import Path

from src.core import get_logger


class SettingsManager:
    """
    Manages application settings, including loading and saving to an INI file.
    """

    def __init__(self, settings_file=None):
        """
        Initialize the settings manager.

        Args:
            settings_file: Path to the settings file. If None, defaults to 'config/settings.ini'
                          in the application directory.
        """
        self.logger = get_logger(__name__)

        # Get the application root directory (parent of src directory)
        self.app_root = Path(__file__).parent.parent.parent

        if settings_file is None:
            self.settings_file = self.app_root / "config/settings.ini"
        else:
            self.settings_file = Path(settings_file)

        # Config parser
        self.config = configparser.ConfigParser()

        # Default settings
        self.default_settings = {
            "Paths": {
                "input_path": "input_folder",
                "output_path": "output_folder",
                "models_path": "models",
                "config_path": "config",
                "log_path": "logs",
                "data_path": "data",
                "docs_path": "docs",
                "db_migrations_path": "src/migrations",
                "tests_path": "src/tests",
            },
            "Default": {
                "device": "cpu",
                "num_workers": "2",
                "batch_size": "4"},
            "Logging": {
                "log_level": "INFO",
                "log_file": "deep_image_app_{datetime}.log",
                "max_bytes": "5242880",
                "backup_count": "3",
            },
        }

        # Load settings
        self.load_settings()

    def load_settings(self):
        """
        Load settings from the settings file. If the file doesn't exist or is invalid,
        use default settings.
        """
        try:
            # Start with default settings
            for section, options in self.default_settings.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                for option, value in options.items():
                    self.config.set(section, option, value)

            # Load from file if it exists
            if self.settings_file.exists():
                self.config.read(self.settings_file)
                self.logger.info(f"Settings loaded from {self.settings_file}")
            else:
                self.logger.info(
                    "Using default settings (no settings file found)")
                
            # Initialize all required directories
            self._initialize_directories()
        except Exception as e:
            self.logger.error(
                f"Error loading settings: {str(e)}. Using defaults.")

    def _initialize_directories(self):
        """
        Initialize all required directories from the Paths section.
        Creates directories if they don't exist.
        """
        try:
            for option in self.default_settings["Paths"]:
                # Get path and resolve it relative to app root if it's not absolute
                path_str = self.get("Paths", option)
                path = Path(path_str)
                if not path.is_absolute():
                    path = self.app_root / path_str
                
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Initialized directory: {path}")
        except Exception as e:
            self.logger.error(f"Error initializing directories: {str(e)}")

    def save_settings(self):
        """
        Save current settings to the settings file.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)

            with open(self.settings_file, "w") as f:
                self.config.write(f)

            self.logger.info(f"Settings saved to {self.settings_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")
            return False

    def get(self, section, option, default=None):
        """
        Get a setting value.

        Args:
            section: The section name.
            option: The option name.
            default: Default value if the option doesn't exist.

        Returns:
            The setting value or the default.
        """
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    def set(self, section, option, value):
        """
        Set a setting value.

        Args:
            section: The section name.
            option: The option name.
            value: The setting value.
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, str(value))

    def get_path(self, section, option, default=None):
        """
        Get a path setting value and ensure it exists.

        Args:
            section: The section name.
            option: The option name.
            default: Default value if the option doesn't exist.

        Returns:
            The path as a Path object.
        """
        path_str = self.get(section, option, default)
        path = Path(path_str)
        os.makedirs(path, exist_ok=True)
        return path

    def reset_to_defaults(self):
        """
        Reset all settings to their default values.
        """
        self.config = configparser.ConfigParser()
        for section, options in self.default_settings.items():
            self.config.add_section(section)
            for option, value in options.items():
                self.config.set(section, option, value)
        self.logger.info("Settings reset to defaults")
