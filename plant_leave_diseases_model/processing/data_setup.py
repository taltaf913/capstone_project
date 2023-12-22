import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path

from plant_leave_diseases_model.config.core import config
from plant_leave_diseases_model import __version__ as _version
from plant_leave_diseases_model.config.core import DATASET_DIR,PACKAGE_ROOT, DATA_STORE_PATH,DATA_STORE_FILE,config,ROOT
import wget
import os
from zipfile import ZipFile
import shutil

output_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH) 
output_data_img_directory = str(output_directory) + "/" +str(config.app_config.dataset_data_dir)

test_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH) + "/TEST"
val_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH) + "/VAL"
train_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH) + "/TRAIN"

class_file_path = str(PACKAGE_ROOT) + "/" + str(config.app_config.dataset_class_dir)+"/"+str(config.app_config.dataset_class_file)

def print_dir():
    print("print_dir:PACKAGE_ROOT:",PACKAGE_ROOT)
    print("print_dir:ROOT:",ROOT)
    
def load_dataset_images():
    print("In data_setup:load_dataset_images")
    print("data_setup:load_dataset_images:config.app_config.dataset_url:",config.app_config.dataset_url,"::PACKAGE_ROOT:",PACKAGE_ROOT)
    url=config.app_config.dataset_url
    #output_directory = DATASET_DIR/"data.zip"
    print("PACKAGE_ROOT:",PACKAGE_ROOT)
    print("DATASET_DIR:",DATASET_DIR)
    print("ROOT::",ROOT)
    
    if not os.path.exists(DATASET_DIR):
        print ("data_setup:load_dataset_images:DATASET_DIR::",DATASET_DIR," NOT EXISTS")
        os.makedirs(DATASET_DIR)
        
    output_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH)
    output_directory_file= str(output_directory) + "/" + str(DATA_STORE_FILE)
    
    print("data_setup:load_dataset_images:output_directory_file:",output_directory_file) 
    #wget {url} -O {download_path}
    wget.download(url,out=output_directory_file)
            
    if os.path.isfile(output_directory_file):
        print("data_setup:load_dataset_images: file downloaded successfully:",output_directory_file)
        # loading the temp.zip and creating a zip object 
        with ZipFile(output_directory_file, 'r') as zObject: 
            print("data_setup:load_dataset_images: extracting file in output_directory::",output_directory)    
            # Extracting all the members of the zip  
            # into a specific location. 
            zObject.extractall( 
                path=output_directory) 
    
            
 
 
def load_leaf_classes():   
    output_directory = str(PACKAGE_ROOT) + "/" + str(DATA_STORE_PATH) 
    output_data_img_directory = str(output_directory) + "/" +str(config.app_config.dataset_data_dir)
    print ("output_data_img_directory::",output_data_img_directory)
    #subfolders = [ f.path for f in os.scandir(output_data_img_directory  ) if f.is_dir() ]
    
    class_file_path = str(PACKAGE_ROOT) + "/" + str(config.app_config.dataset_class_dir)+"/"+str(config.app_config.dataset_class_file)
    print ("data_setup:load_leaf_classes:class_file",class_file_path)
    print("################################################")
    #reverse_sorted_items = sorted(os.scandir(output_data_img_directory), reverse=True)
    reverse_sorted_items = sorted_directory_listing(output_data_img_directory)
    print ("data_setup:load_leaf_classes:reverse_sorted_items:",reverse_sorted_items)
    print("################################################")
    if os.path.exists(class_file_path):
        print("data_setup:load_leaf_classes: file exists, hence deleting")
        os.remove(class_file_path)
    else:
        print("The file does not exist:",class_file_path)
    for f in reverse_sorted_items:
        print("data_setup:load_leaf_classes:f:",f)
        with open(class_file_path, 'a') as the_file:
            the_file.write(str(f)+"\n")
    
    
    return  class_file_path    
           
            
def sorted_directory_listing(directory):
    items = os.listdir(directory)
    reverse_sorted_items = sorted(items, reverse=False)
    return reverse_sorted_items    


def prepare_data_images_per_class():
    print("data_setup::prepare_data_images_per_class::start")
    text_file = open(class_file_path, "r")
    img_classes = text_file.readlines()
    print (img_classes)
    print ("data_setup:prepare_data_images_per_class:length of classes:",len(img_classes))
    text_file.close()
    
    for i in range(config.model_config.no_of_classes):
        img_class=img_classes[i].strip("\n")
        print("img_class::",img_class)
        file_dir=str(output_data_img_directory) + "/" +str(img_class)
        class_files_arr = [ f.path for f in os.scandir(file_dir ) if f.is_file() ]
        print("img_class:",img_class,"::class_files_arr:len:",len(class_files_arr))
        #preparing train data per class
        dest_directory=str(train_directory)+"/"+str(img_class)
        copy_class_images(0,
                          config.model_config.no_of_img_per_class_train,
                          class_files_arr,
                          img_class,
                          dest_directory)
        
        #preparing val data per class
        dest_directory=str(val_directory)+"/"+str(img_class)
        copy_class_images(config.model_config.no_of_img_per_class_train,
                          config.model_config.no_of_img_per_class_train+config.model_config.no_of_img_per_class_val,
                          class_files_arr,
                          img_class,
                          dest_directory)

        #preparing test data per class
        dest_directory=str(test_directory)+"/"+str(img_class)
        copy_class_images(config.model_config.no_of_img_per_class_train + config.model_config.no_of_img_per_class_val,
                          config.model_config.no_of_img_per_class_train+config.model_config.no_of_img_per_class_val+config.model_config.no_of_img_per_class_test,
                          class_files_arr,
                          img_class,
                          dest_directory)
          
        
        #for img_fil in class_files_arr:
            
def copy_class_images(start,no_of_img_per_class,class_files_arr,img_class,dest_directory):
    print("data_setup:copy_class_images")
    print("start:",start)
    print("no_of_img_per_class:",no_of_img_per_class)
    print("class_files_arr:",len(class_files_arr))
    print("img_class:",img_class)
    print("dest_directory:",dest_directory)
    for i in range(start,no_of_img_per_class):
        img_trn_file=class_files_arr[i]
        dest_img_trn_file_arr=img_trn_file.split("/")
        dest_img_trn_file=dest_img_trn_file_arr[-1:][0]
        dest_class_dir=str(dest_directory)
        print("img_class:",img_class,"::img file name:",img_trn_file,"::dest_img_trn_file:",dest_img_trn_file)
        if not os.path.exists(dest_class_dir):
            print ("data_setup:prepare_data_images_per_class:dest_class_dir::",dest_class_dir," NOT EXISTS, hence creating it")
            os.makedirs(dest_class_dir)
        shutil.copy2(img_trn_file, dest_directory+"/"+dest_img_trn_file)
           
    
     
