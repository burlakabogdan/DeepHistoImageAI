import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
from sqlalchemy.orm import Session

from src.core.models import DeepLearningModel


class ModelManager:
    """Manages deep learning models and their metadata."""

    def __init__(self, db_session: Session, models_path: str, device: str = None):
        """
        Initialize ModelManager.

        Args:
            db_session: SQLAlchemy database session
            models_path: Path to models directory
        """
        self.db_session = db_session
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.current_model = None   
        self.device = device if device and device.strip() else self._get_device()
        self.logger.info(f"ModelManager initialized with device: {self.device}")
        # print(f"ModelManager initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Get available device with fallback to CPU."""
        if torch.cuda.is_available():
            try:
                # Test CUDA availability
                torch.cuda.empty_cache()
                torch.zeros(1).cuda()
                return "cuda"
            except Exception as e:
                self.logger.warning(
                    f"CUDA available but error occurred: {e}. Falling back to CPU.")
                return "cpu"
        return "cpu"

    def load_model(self, model_guid: str) -> Tuple[bool, str]:
        """
        Load a model by its GUID.

        Args:
            model_guid: Model GUID

        Returns:
            Tuple[bool, str]: (success, message)
        """
        self.logger.info(f"Loading model to device: {self.device}")
        try:
            # Get model data from database
            model_data = (
                self.db_session.query(DeepLearningModel).filter_by(
                    model_guid=model_guid).first())

            if not model_data:
                return False, f"Model with GUID {model_guid} not found in database"

            # Construct model path
            model_path = self.models_path / model_guid / model_data.model_file_name

            # Check if model file exists
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"

            self.logger.info(
                f"Loading model {model_data.model_name} from {model_path}")
            self.logger.info(
                f"Model details: architecture={model_data.model_architecture}, "
                f"encoder={model_data.model_encoder}, classes={model_data.model_num_classes}")

            # Initialize model architecture
            model = self._initialize_model(
                architecture=model_data.model_architecture,
                encoder=model_data.model_encoder,
                encoder_weights=model_data.model_weights,
                in_channels=model_data.model_in_channels,
                classes=model_data.model_num_classes,
            )

            # Load model weights
            self.logger.info(f"Loading model weights from: {model_path}")
            try:
                # First try loading the state dict directly
                state_dict = torch.load(model_path, map_location=self.device)

                # Debug state dict
                self.logger.info(
                    f"State dict keys: {list(state_dict.keys())[:5]}... (showing first 5)"
                )

                # Check if the state dict matches the expected format
                if any(key.startswith("arc.") for key in state_dict.keys()):
                    self.logger.info(
                        "State dict has keys with 'arc.' prefix, seems to be for a wrapped model"
                    )
                    load_result = model.load_state_dict(
                        state_dict, strict=False)
                else:
                    self.logger.info(
                        "State dict doesn't have 'arc.' prefix, trying to adapt it")
                    # Try to adapt the state dict for our wrapped model
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_state_dict[f"arc.{key}"] = value

                    load_result = model.load_state_dict(
                        new_state_dict, strict=False)

                # Log any missing or unexpected keys
                if load_result.missing_keys:
                    self.logger.warning(
                        f"Missing key(s) in state_dict: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    self.logger.warning(
                        f"Unexpected key(s) in state_dict: {load_result.unexpected_keys}"
                    )
            except Exception as e:
                self.logger.error(f"Error loading state dict: {str(e)}")
                # Try alternative loading approach
                try:
                    self.logger.info("Trying alternative loading approach...")
                    # Try loading the model directly without wrapping
                    direct_model = self._initialize_direct_model(
                        architecture=model_data.model_architecture,
                        encoder=model_data.model_encoder,
                        encoder_weights=model_data.model_weights,
                        in_channels=model_data.model_in_channels,
                        classes=model_data.model_num_classes,
                    )

                    state_dict = torch.load(
                        model_path, map_location=self.device)
                    direct_model.load_state_dict(state_dict, strict=False)

                    # Now wrap the loaded model
                    class SegmentationModel(torch.nn.Module):
                        def __init__(self, arc):
                            super(SegmentationModel, self).__init__()
                            self.arc = arc

                        def forward(self, images, masks=None):
                            logits = self.arc(images)
                            return logits

                    model = SegmentationModel(direct_model)
                    self.logger.info("Alternative loading approach succeeded")
                except Exception as e2:
                    self.logger.error(
                        f"Alternative loading also failed: {str(e2)}")
                    return (
                        False,
                        f"Error loading model weights: {str(e)}, alternative approach also failed: {str(e2)}",
                    )

            # model.to(self.device)
            model.to(torch.device(self.device))
            # Debug: print device info
            self.logger.info(f"Model moved to device: {self.device}")
            
            model.eval()  # Set to evaluation mode

            self.current_model = {
                "model": model,
                "metadata": model_data,
                "guid": model_guid,
            }

            self.logger.info(
                f"Model {model_data.model_name} loaded successfully")
            return True, "Model loaded successfully"

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False, f"Error loading model: {str(e)}"

    def _initialize_model(
        self,
        architecture: str,
        encoder: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
    ) -> torch.nn.Module:
        """Initialize model architecture."""
        try:
            self.logger.info(
                f"Initializing model with architecture={architecture}, encoder={encoder}, "
                f"encoder_weights={encoder_weights}, in_channels={in_channels}, classes={classes}")

            if hasattr(smp, architecture):
                model_class = getattr(smp, architecture)

                # Create the base model with activation=None for proper logits
                # output
                base_model = model_class(
                    encoder_name=encoder,
                    encoder_weights=encoder_weights,
                    in_channels=in_channels,
                    classes=classes,
                    activation=None,  # Explicitly set activation to None for proper logits output
                )

                # Wrap the model in a SegmentationModel class for consistent
                # interface
                class SegmentationModel(torch.nn.Module):
                    def __init__(self, arc):
                        super(SegmentationModel, self).__init__()
                        self.arc = arc

                    def forward(self, images, masks=None):
                        logits = self.arc(images)
                        return logits

                model = SegmentationModel(base_model)
                self.logger.info(
                    f"Model initialized successfully: {type(model).__name__} with {type(base_model).__name__}"
                )
                return model
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _initialize_direct_model(
        self,
        architecture: str,
        encoder: str,
        encoder_weights: str,
        in_channels: int,
        classes: int,
    ) -> torch.nn.Module:
        """Initialize model architecture without wrapping."""
        if hasattr(smp, architecture):
            model_class = getattr(smp, architecture)
            return model_class(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=None,  # Explicitly set activation to None for proper logits output
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def validate_model_compatibility(
            self, description_path: Path) -> Tuple[bool, str]:
        """
        Validate model compatibility from description.json.

        ##TODO do more validations of models

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            with open(description_path) as f:
                description = json.load(f)

            # Validate required fields
            required_fields = [
                "model_architecture",
                "model_encoder",
                "model_in_channels",
                "model_num_classes",
                "image_size",
            ]

            missing_fields = [
                field for field in required_fields if field not in description]

            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"

            # Validate architecture
            if not hasattr(smp, description["model_architecture"]):
                return (
                    False,
                    f"Unsupported architecture: {description['model_architecture']}",
                )

            # Try initializing model
            try:
                self._initialize_model(
                    architecture=description["model_architecture"],
                    encoder=description["model_encoder"],
                    encoder_weights=description.get("model_weights", None),
                    in_channels=description["model_in_channels"],
                    classes=description["model_num_classes"],
                )
            except Exception as e:
                return False, f"Error initializing model: {str(e)}"

            return True, "Model is compatible"

        except Exception as e:
            return False, f"Error validating model: {str(e)}"

    def get_all_models(self):
        """
        Get all models from the database.

        Returns:
            List of DeepLearningModel objects
        """
        if not self.db_session:
            self.logger.warning("No database session available")
            return []

        try:
            return self.db_session.query(DeepLearningModel).all()
        except Exception as e:
            self.logger.error(f"Error retrieving models: {str(e)}")
            return []

    def get_current_model(self) -> Optional[Dict]:
        """Get currently loaded model and its metadata."""
        return self.current_model

    def unload_model(self) -> None:
        """Unload current model and clear CUDA memory."""
        if self.current_model is not None:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self.current_model = None

    def register_model(
        self,
        model_guid: str,
        model_name: str,
        model_file_name: str,
        model_description: str = "",
        model_architecture: str = "Unet",
        model_encoder: str = "resnet34",
        model_weights: str = "imagenet",
        model_in_channels: int = 3,
        model_num_classes: int = 2,
        model_class_names: json = "",
        model_class_colours: json = "",
        staining: str = "none",
        image_size: int = 512,
        normalization_method: str = "none",
        stain_normalization_method: str = None,
        training_image_example: str = None,
        training_image_mask_example: str = None,
        normalization_image: str = None,
        model_version: str = "1.0.0",
        create_date: datetime = None,
        last_modified: datetime = None,
    ) -> Tuple[bool, str]:
        """
        Register a model in the database.

        Args:
            model_guid: Model GUID
            model_name: Model name
            model_file_name: Model file name
            model_architecture: Model architecture (default: 'Unet')
            model_encoder: Model encoder (default: 'resnet34')
            model_weights: Model weights (default: 'imagenet')
            model_in_channels: Number of input channels (default: 3)
            model_num_classes: Number of output classes (default: 2)
            normalization_method: Normalization method (default: 'none')
            stain_normalization_method: Stain normalization method (default: None)
            normalization_image: Normalization image path (default: None)
            image_size: Image size for inference (default: 512)
            training_image_example: Training image example path (default: None)
            training_image_mask_example: Training image mask example path (default: None)
            model_version: Model version (default: '1.0.0')
            create_date: Creation date (default: None)
            last_modified: Last modified date (default: None)

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Check if model already exists
            existing_model = (
                self.db_session.query(DeepLearningModel).filter_by(
                    model_guid=model_guid).first())
            if existing_model:
                return False, f"Model with GUID {model_guid} already exists"

            # Create new model entry
            model = DeepLearningModel(
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
                normalization_image=normalization_image,
                training_image_example=training_image_example,
                training_image_mask_example=training_image_mask_example,
                model_version=model_version,
                create_date=create_date,
                last_modified=last_modified,
            )

            # Add to database
            self.db_session.add(model)
            self.db_session.commit()

            self.logger.info(f"Model {model_name} registered successfully")
            return True, "Model registered successfully"

        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.db_session.rollback()
            return False, f"Error registering model: {str(e)}"
