from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DeepLearningModel(Base):
    __tablename__ = "deep_learning_models"

    id = Column(Integer, primary_key=True)
    model_guid = Column(String, unique=True, nullable=False)
    model_name = Column(String, nullable=False)
    model_file_name = Column(String, nullable=False)
    model_description = Column(String)
    model_architecture = Column(String, nullable=False)
    model_encoder = Column(String, nullable=False)
    model_weights = Column(String)
    model_in_channels = Column(Integer, nullable=False)
    model_num_classes = Column(Integer, nullable=False)
    model_class_names = Column(JSON)
    model_class_colours = Column(JSON)
    staining = Column(String)
    image_size = Column(Integer, nullable=False)
    # 'none', 'imagenet', 'custom', 'stain'
    normalization_method = Column(String)
    # 'macenko', 'reinhard', 'reinhard_modified'
    stain_normalization_method = Column(String)
    training_image_example = Column(String)
    training_image_mask_example = Column(String)
    normalization_image = Column(String)
    model_version = Column(String)
    create_date = Column(DateTime, default=datetime.utcnow)
    last_modified = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow)


# Database initialization function
def init_db(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return engine
