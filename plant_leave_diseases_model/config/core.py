# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from strictyaml import YAML, load

import plant_leave_diseases_model

# Project Directories
PACKAGE_ROOT = Path(plant_leave_diseases_model.__file__).resolve().parent  
ROOT = PACKAGE_ROOT.parent.parent
arr=str(PACKAGE_ROOT).split("/")
print("CORE:ROOT:",ROOT)
MODEL_NAME=arr[-1:][0]
print("CORE:MODEL_NAME:",MODEL_NAME)
#PACKAGE_ROOT= ROOT / MODEL_NAME
print("CORE:PACKAGE_ROOT:",PACKAGE_ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets/data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DOWNLOAD_PATH = "plant_leaf_diseases_dataset_with_augmentation"  

DATA_STORE_PATH = "datasets/data"
DATA_STORE_FILE = "data.zip"




class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    train_path: str
    validation_path: str
    test_path: str
    model_name: str
    model_save_file: str
    dataset_url: str
    #download_path: str
    dataset_data_dir: str
    dataset_class_dir: str
    dataset_class_file: str

class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """
        
    image_size: List[int]
    batch_size: int
    scaling_factor: float
    rotation: float
    zoom: float
    flip: str

    random_state: int
    input_shape: List[int]
    epochs: int
    optimizer: str
    loss: str
    accuracy_metric: str
    verbose: int
    earlystop: int
    monitor: str
    save_best_only: bool
    label_mappings: Dict[int, str]
    learning_rate: float
    no_of_classes: int
    no_of_img_per_class_test: int
    no_of_img_per_class_train: int
    no_of_img_per_class_val: int
    #leaf_class_category_mappings: Dict[str, int]
    leaf_class_master_category_list: List[str]
    # MLFLow parameter
    run_name : str
    
class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
        
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config = AppConfig(**parsed_config.data),
        model_config = ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()