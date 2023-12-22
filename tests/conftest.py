import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import warnings
warnings.filterwarnings("ignore")
from plant_leave_diseases_model.processing.data_manager import load_leaf_disease_dataset, load_test_dataset
from plant_leave_diseases_model.processing.data_setup import load_dataset_images, load_leaf_classes, prepare_data_images_per_class, print_dir, test_directory,output_data_img_directory
import os

@pytest.fixture
def sample_input_data():
    #Image download and Setup
    print_dir()
    if os.path.exists(output_data_img_directory):
        print("Image repo is present in : ",output_data_img_directory,": no need to  download it again")
    else:
        
        print(print("Image repo is NOT present in : ",output_data_img_directory,": proceeding to download it "))
        load_dataset_images()
        load_leaf_classes()
        prepare_data_images_per_class()
    
    ##########################################
    # Get train data : x_train, y_train
    ##########################################
    x_test, y_test,leaf_disease_master_classes = load_leaf_disease_dataset(test_directory)

    print("conf_test:sample_input_data:x_train_size:",len(x_test))
    print("conf_test:sample_input_data:y_train_size:",len(y_test))
    print("conf_test:sample_input_data:x_train.shape:",x_test.shape)
    print("conf_test:sample_input_data:y_train.shape:",y_test.shape)

    return x_test, y_test