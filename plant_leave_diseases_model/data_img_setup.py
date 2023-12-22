import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from plant_leave_diseases_model.processing.data_setup import load_dataset_images, print_dir

if __name__ == "__main__":
    print_dir()
    #load_dataset_images()
    #load_leaf_classes()
    #prepare_data_images_per_class()