import sys
from pathlib import Path
import mlflow

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from plant_leave_diseases_model.config.core import config,PACKAGE_ROOT,TRAINED_MODEL_DIR
from plant_leave_diseases_model.model import create_model
from plant_leave_diseases_model.processing.data_manager import get_one_hot_data_for_input_classes, load_leaf_disease_dataset, load_train_dataset, load_validation_dataset, load_test_dataset, callbacks_and_save_model,prepare_img_data,get_class_file_list,get_model_file_name_path,get_master_classes_in_data_frame
from plant_leave_diseases_model.processing.data_setup import load_dataset_images, load_leaf_classes, prepare_data_images_per_class, print_dir, test_directory,val_directory,train_directory,class_file_path,output_data_img_directory

from sklearn.preprocessing import LabelBinarizer
import cv2
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import pickle
from sklearn.preprocessing import OneHotEncoder
from plant_leave_diseases_model import __version__ as _version 
#from plant_leave_diseases_model.predict import make_prediction
import os
default_image_size = tuple((256, 256))
    
    
def run_training() -> None:
    ################################
    # Init MLFlow experiment tracking
    ################################
    mlflow.set_tracking_uri(f"http://54.89.245.21:5000")
    exp = mlflow.set_experiment(experiment_name = "Plant leaf disease Experiments")
    
    mlflow.start_run(run_name= config.model_config.run_name, experiment_id= exp.experiment_id)
    # Log parameters
    mlflow.log_param("Plant classes/category", config.model_config.no_of_classes)
    mlflow.log_param("Training size per category", config.model_config.no_of_img_per_class_train)
    mlflow.log_param("Testing size per category", config.model_config.no_of_img_per_class_test)
    mlflow.log_param("Validation size per category", config.model_config.no_of_img_per_class_val)
    mlflow.log_param("epochs", config.model_config.epochs)
    mlflow.log_param("batch size", config.model_config.batch_size)
    
    """
    Train the model.
    """
    
    ##########################################
    # Get train data : x_train, y_train
    ##########################################
    x_train, y_train,leaf_disease_master_classes = load_leaf_disease_dataset(train_directory)

    print("x_train_size:",len(x_train))
    print("y_train_size:",len(y_train))
    print("x_train.shape:",x_train.shape)
    print("y_train.shape:",y_train.shape)
    
    ##########################################
    # Get validation data : x_val, y_val
    ##########################################
    x_val, y_val,leaf_disease_master_classes = load_leaf_disease_dataset(val_directory)

    print("x_val_size:",len(x_val))
    print("y_val_size:",len(y_val))
    print("x_val.shape:",x_val.shape)
    print("y_val.shape:",y_val.shape)
    
    ##############################################################
    # Getting no of classes to pass thw model at last layer
    ##############################################################
    n_classes = len(leaf_disease_master_classes)
    
    ################################
    # Create model
    ################################
    model = create_model(input_shape = config.model_config.input_shape, 
                          optimizer = config.model_config.optimizer, 
                          loss = config.model_config.loss, 
                          metrics = [config.model_config.accuracy_metric],
                          n_classes = n_classes
                        )
    ################################
    # Training the model
    ################################
  
    history = model.fit(
    x_train, 
    y_train, 
    batch_size=config.model_config.batch_size,
    validation_data=(x_val, y_val),
    #steps_per_epoch=len(x_train) // 5,
    #callbacks = callbacks_and_save_model(),
    epochs=config.model_config.epochs
    )

    ################################
    # Saving the model
    ################################
    save_model_file_name = get_model_file_name_path()
    print("###################### ####### ##########################")
    print("save_model_file_name:",save_model_file_name)
    print("###################### ####### ##########################")

    model.save(save_model_file_name)
    
    ##########################################
    # Get test data : x_test, y_test
    ##########################################
    
    x_test, y_test,leaf_disease_master_classes = load_leaf_disease_dataset(test_directory)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("test_accuracy:test_accuracy:(test_loss,test_acc):",test_loss,",",test_acc)
    
    ##########################################
    # Log MLFlow performance metrics
    ##########################################
    
    mlflow.log_metric("Model accuracy", test_acc)
    mlflow.log_metric("Model loss", test_loss)
    
    #mlflow.sklearn.log_model(sk_model = rf, artifact_path= "trained_model")
    mlflow.end_run()
     
    
if __name__ == "__main__":
    print_dir()
    if os.path.exists(output_data_img_directory):
        print("Image repo is present in : ",output_data_img_directory,": no need to  download it again")
    else:
        
        print(print("Image repo is NOT present in : ",output_data_img_directory,": proceeding to download it "))
        load_dataset_images()
        load_leaf_classes()
        prepare_data_images_per_class()
    
    print("running trainning ")
    run_training()
    # Define directory where test images are loaded
    print("###################### ####### ##########################")
    print("###################### PREDICT ##########################")
    print("###################### ####### ##########################")

    #test_dir_img_file_path=test_directory+"/Apple___Apple_scab"
    #print("test_dir_img_file_path::",test_dir_img_file_path) 
    #make_prediction(test_dir_img_file_path = test_dir_img_file_path)
