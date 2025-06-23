import logging
import os
from pathlib import Path
from typing import List, Tuple


class FileManager:
    """Manages file system operations for the Deep Image application."""

    ALLOWED_EXTENSIONS = {".tif", ".jpeg", ".png"}

    def __init__(self, input_path: str, output_path: str):
        """Initialize FileManager with input and output paths."""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.logger = logging.getLogger(__name__)

        # Create directories if they don't exist
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_input_files(self) -> List[Path]:
        """Get list of valid image files from input directory."""
        files = []
        for ext in self.ALLOWED_EXTENSIONS:
            files.extend(self.input_path.glob(f"*{ext}"))
        return sorted(files)

    def validate_file_extension(self, filepath: Path) -> bool:
        """Validate if file has allowed extension."""
        return filepath.suffix.lower() in self.ALLOWED_EXTENSIONS

    def get_output_path(self, input_file: Path) -> Path:
        """Generate output path for mask file."""
        return self.output_path / f"{input_file.stem}.tif"

    def ensure_output_directory(self) -> None:
        """Ensure output directory exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def cleanup_temp_files(self, temp_dir: Path) -> None:
        """Clean up temporary files."""
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        import shutil

                        shutil.rmtree(file)
                except Exception as e:
                    self.logger.error(f"Error cleaning up {file}: {str(e)}")
            try:
                temp_dir.rmdir()
            except Exception as e:
                self.logger.error(f"Error removing temp directory: {str(e)}")

    # TODO remove it
    # def import_model_files(self, archive_path: Path, temp_dir: Path, model_dir: Path) -> Tuple[bool, str]:
    #     """
    #     Import model files from archive to model directory.
    #     Returns (success: bool, message: str)
    #     """
    #     try:
    #         import zipfile

    #         # Clear and create temp directory
    #         self.cleanup_temp_files(temp_dir)
    #         temp_dir.mkdir(parents=True, exist_ok=True)

    #         # Extract archive
    #         with zipfile.ZipFile(archive_path, 'r') as zip_ref:
    #             zip_ref.extractall(temp_dir)

    #         # Verify required files
    #         required_files = {'model_file': False, 'description.json': False,
    #                         'image_example': False, 'mask_example': False}

    #         for file in temp_dir.glob('**/*'):
    #             if file.suffix in {'.pt', '.pth'}:
    #                 required_files['model_file'] = True
    #             elif file.name == 'description.json':
    #                 required_files['description.json'] = True
    #             elif any(file.suffix.lower() in self.ALLOWED_EXTENSIONS for ext in ['.tif', '.jpeg', '.png']):
    #                 if 'mask' in file.stem.lower():
    #                     required_files['mask_example'] = True
    #                 else:
    #                     required_files['image_example'] = True

    #         if not all(required_files.values()):
    #             missing = [k for k, v in required_files.items() if not v]
    #             return False, f"Missing required files: {', '.join(missing)}"

    #         # Copy files to model directory
    #         model_dir.mkdir(parents=True, exist_ok=True)
    #         import shutil
    #         for file in temp_dir.glob('**/*'):
    #             if file.is_file():
    #                 shutil.copy2(file, model_dir / file.name)

    #         return True, "Model files imported successfully"

    #     except Exception as e:
    #         return False, f"Error importing model files: {str(e)}"
    #     finally:
    #         self.cleanup_temp_files(temp_dir)

    def import_model_files(
        self, archive_path: Path, temp_dir: Path, model_dir: Path
    ) -> Tuple[bool, str]:
        """
        Import model files from archive to model directory.
        Returns (success: bool, message: str)
        """
        try:
            import json
            import zipfile

            # Clear and create temp directory
            self.cleanup_temp_files(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Extract archive
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Initialize required files tracking
            required_files = {
                "model_file": False,
                "description.json": False,
                "training_image_example": False,
                "training_image_mask_example": False,
            }

            # First check if description.json exists
            description_file = next((file for file in temp_dir.glob(
                "**/*") if file.name == "description.json"), None, )
            if not description_file:
                return False, "Missing required file: description.json"

            required_files["description.json"] = True

            # Read description.json to get image and mask example filenames
            try:
                with open(description_file, "r") as f:
                    description_data = json.load(f)

                image_example_name = description_data.get(
                    "training_image_example")
                mask_example_name = description_data.get(
                    "training_image_mask_example")

                if not image_example_name or not mask_example_name:
                    return (
                        False,
                        "Missing training_image_example or training_image_mask_example fields in description.json",
                    )
            except Exception as e:
                return False, f"Error reading description.json: {str(e)}"

            # Check for model file
            model_file = next((file for file in temp_dir.glob(
                "**/*") if file.suffix in {".pt", ".pth"}), None, )
            if model_file:
                required_files["model_file"] = True

            # Check for image and mask examples using the names from
            # description.json
            image_example_file = next((file for file in temp_dir.glob(
                "**/*") if file.name == image_example_name), None, )
            if image_example_file and image_example_file.suffix.lower() in self.ALLOWED_EXTENSIONS:
                required_files["training_image_example"] = True

            mask_example_file = next((file for file in temp_dir.glob(
                "**/*") if file.name == mask_example_name), None, )
            if mask_example_file and mask_example_file.suffix.lower() in self.ALLOWED_EXTENSIONS:
                required_files["training_image_mask_example"] = True

            # Check if all required files are present
            if not all(required_files.values()):
                missing = [k for k, v in required_files.items() if not v]
                return False, f"Missing required files: {', '.join(missing)}"

            # Copy files to model directory
            model_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            for file in temp_dir.glob("**/*"):
                if file.is_file():
                    shutil.copy2(file, model_dir / file.name)

            return True, "Model files imported successfully"

        except Exception as e:
            return False, f"Error importing model files: {str(e)}"
        finally:
            self.cleanup_temp_files(temp_dir)
