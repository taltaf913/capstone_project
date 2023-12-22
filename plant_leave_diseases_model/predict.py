from os import listdir
import sys
from pathlib import Path

import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Union
import pandas as pd
import tensorflow as tf

from plant_leave_diseases_model import __version__ as _version
from plant_leave_diseases_model.config.core import TRAINED_MODEL_DIR, config
from plant_leave_diseases_model.processing.data_manager import convert_image_to_array, get_class_file_list, get_model_file_name_path, load_leaf_disease_dataset, load_model, load_test_dataset
from plant_leave_diseases_model.processing.data_setup import test_directory,class_file_path


def make_prediction(*, test_dir_img_file_path) -> dict:
    """Make a prediction using a saved model """
    image_list=[]
    
    #x_test, y_test = load_leaf_disease_dataset(test_dir_img_file_path)
    root_dir = listdir(test_dir_img_file_path)
    for plant_image in root_dir :
        print("plant_image::",plant_image)
        img_file_path=str(test_dir_img_file_path)+"/"+str(plant_image)
        if plant_image.endswith(".jpg") == True or plant_image.endswith(".JPG") == True:
            image_list.append(convert_image_to_array(img_file_path))
            
    np_image_list = np.array(image_list, dtype=np.float16) / config.model_config.scaling_factor
           
    print("np_image_list.shape:",np_image_list.shape)
    results = {"predictions": None, "version": _version}
    
    load_model_and_predict(np_image_list) 
    pred_labels = model_file_name = get_model_file_name_path()
        
    results = {"predictions": pred_labels, "version": _version}
    

    return results

def load_model_and_predict(x_test):
    model_file_name = get_model_file_name_path()
    
    print("loading mode file:",model_file_name)
    
    model = load_model(file_name = model_file_name)
    
    predictions = model.predict(x_test,verbose = 0)
    
    ###########################################
    # Geting array of master class array
    ###########################################
    master_class_arr=get_class_file_list(class_file_path)
    print("master_class_arr:",master_class_arr)
    
    print(predictions)
    pred_labels = []
    for pred in predictions:
        np.multiply(np.array(pred), 100000)
        max_index = np.multiply(np.array(pred), 100000).argmax()
        leaf_pred_label=master_class_arr[max_index]
        print("load_model_and_predict:max_index::",max_index,"master_class_arr:",master_class_arr[max_index])
        pred_labels.append(leaf_pred_label)
        
    return pred_labels    

def get_test_accuracy():
    ##########################################
    # Get validation data : x_val, y_val
    ##########################################
    x_test, y_test,leaf_disease_master_classes = load_leaf_disease_dataset(test_directory)

    print("x_test_size:",len(x_test))
    print("y_test_size:",len(y_test))
    print("x_test.shape:",x_test.shape)
    print("y_test.shape:",y_test.shape)
    
    model_file_name = get_model_file_name_path()
    model = load_model(file_name = model_file_name)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("predict:get_test_accuracy:test_accuracy:(test_loss,test_acc):",test_loss,",",test_acc)


if __name__ == "__main__":

    # Define directory where test images are loaded
    test_dir_img_file_path=test_directory+"/Apple___Apple_scab"
    print("test_dir_img_file_path::",test_dir_img_file_path) 
    make_prediction(test_dir_img_file_path = test_dir_img_file_path)
    get_test_accuracy()
