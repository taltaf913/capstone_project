import gradio as gr
import openai
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import requests
import json
from json import JSONEncoder
from datetime import datetime
import requests
from os import listdir
import json
import os
import cv2
import joblib
from json import JSONEncoder
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime

from sklearn.preprocessing import LabelBinarizer
import sklearn.preprocessing
import pickle
from sklearn.preprocessing import OneHotEncoder
import cv2
from keras.preprocessing.image import img_to_array, array_to_img
from fastapi import FastAPI, Request, Response
from sklearn.metrics import precision_score, recall_score, f1_score
import prometheus_client as prom
# Load your trained model
app = FastAPI()
DEFAULT_IMAGE_SIZE = tuple((256, 256))
N_IMAGES = 100
# Define the model and its parameters
model = tf.keras.models.load_model('cnn_model_v3.h5')
test_directory = "/etc/TEST"
#test_dir_img_file_path = os.path.getcwd("imagesfolder")
#test_dir_img_file_path = os.path.join(os.getcwd(),"imagesfolder")
#test_dir_img_file_path = test_directory+"/imagesfolder"
# Define the metrics
precision_metric = prom.Gauge("plant_precision", "precision score for random 50 test samples")
recall_metric = prom.Gauge("plant_recall_score", "recall score for random 50 test samples")
f1_metric = prom.Gauge("plant_f1_score", "F1 score for random 50 test samples")


def convert_image_to_array(image_dir):
    try:
        print(image_dir)
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
# Define the function to update the metrics
image_list = []
label_list = []
def update_metrics():
    # Load the test images
    root_dir = listdir(test_directory)
    for plant_folder in root_dir :
        if not plant_folder.startswith('.DS_Store'):
            plant_disease_folder_list = listdir(f"{test_directory}/{plant_folder}")
            print(f"{test_directory}/{plant_folder}")
            for plant_disease_folder in plant_disease_folder_list:
               if not plant_disease_folder.startswith('.DS_Store'):
                    image_directory = (f"{test_directory}/{plant_folder}/{plant_disease_folder}")
                    print(image_directory)
                    try :

                        if os.path.isfile(image_directory) and (image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True):
                           image_list.append(convert_image_to_array(image_directory))
                           label_list.append(plant_folder)
                    except :
                        pass

    np_image_list = np.array(image_list, dtype=np.float16) / 256.0
    #print("lables")
    #print(label_list)

    print("np_image_list.shape:",np_image_list.shape)
    prediction = model.predict(np_image_list)
    leaf_class_category_mappings1 = {0:"Apple Apple scab", 1:"Apple Black rot", 2:"Apple Cedar apple rust",3:"Apple healthy", 4:"Background without leaves", 5:"Blueberry healthy", 6:"Cherry Powdery mildew", 7:"Cherry healthy",8:"Corn Cercospora leaf spotGray leaf spot", 9:"Corn Common rust", 10:"Corn Northern Leaf Blight", 11:"Corn healthy",12:"Grape Black rot", 13:"Grape Esca (Black Measles)", 14:"Grape Leaf blight (Isariopsis Leaf Spot)", 15:"Grape healthy",16:"Orange Haunglongbing (Citrus greening)", 17:"Peach Bacterial spot", 18:"Peach healthy",19:"Pepper  bell Bacterial spot",20:"Pepper  bell healthy", 21:"Potato Early blight", 22:"Potato Late blight", 23:"Potato healthy", 24:"Raspberry healthy",25:"Soybean healthy", 26:"Squash Powdery mildew", 27:"Strawberry Leaf scorch", 28:"Strawberry healthy", 29:"Tomato Bacterial spot",30:"Tomato Early blight", 31:"Tomato Late blight", 32:"Tomato Leaf Mold", 33:"Tomato Septoria leaf spot", 34:"Tomato Spider mitesTwo-spotted spider mite",35:"Tomato Target Spot", 36:"Tomato Tomato Yellow Leaf Curl Virus", 37:"Tomato Tomato mosaic virus", 38:"Tomato healthy"}
    #return ["leaf_class_category_mappings[np.argmax(prediction)]"
    #return [leaf_class_category_mappings1[np.argmax(predictions)], "none"]
    print(prediction)
    # Make predictions

    # Calculate the metrics
    #pred = np.argmax(prediction, axis=1)
    print(np.argmax(prediction, axis=1).shape)
    #pred = leaf_class_category_mappings1[print(np.argmax(prediction, axis=1))]
    print(list(np.argmax(prediction, axis=1)))
    #print(np.argmax(prediction))
    y_true = [label.replace("___", " ").replace("_", " ") for label in label_list]
    print(y_true)
    pred = list(np.argmax(prediction, axis=1))
    pred = [leaf_class_category_mappings1[i] for i in pred]
    print(pred)
    f1 = f1_score(pred, y_true, average='macro')
    precision = precision_score(pred, y_true, average='macro')
    recall = recall_score(pred, y_true, average='macro')
    print("f1:", f1)
    #test_acc = model.evaluate(image_list, label_list)
    #print(f"Test Accuracy: {test_acc[1]*100}")
    #print("test_accuracy:test_accuracy:(test_loss,test_acc):",test_loss,",",test_acc)

    # Update the metrics
    f1_metric.set(f1)
    precision_metric.set(precision)
    recall_metric.set(recall)


@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())



myControls = {
    "ResultControl":None,
    "Feedback":None,
    "AdditionalInfo":None
}


dataToSend = {
    "FileContent":None,
    "PlantName":None,
    "Comments":None
}

imageData = []

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) :
            return obj.tolist()
        return JSONEncoder.default(self,obj)



def uploadFile() :
    global dataToSend

    npArray = np.asarray(imageData)
    fileContent = json.dumps(npArray, cls=NumpyEncoder)
    dataToSend["FileContent"] = fileContent

    payload = json.dumps(dataToSend)


    r = requests.post("http://54.89.245.21:5005/todb", data=payload)
    return r

def saveStats(predictionStatus) :
    d = {
        'Time': str(datetime.now()),
        'PredictionStatus':None
    }

    if predictionStatus == 'Satisfied' :
        d['PredictionStatus'] = 1
    else :
        d['PredictionStatus'] = 0

    r = requests.post("http://54.89.245.21:5005/predictionstats", data=json.dumps(d))
    return r


def predict(imageToProcess):
    global imageData
    imagelist = []
    if imageToProcess == None :
        gr.Error("No image is given to process")
        return [None, None]

    imageToProcess.save("/tmp/test.jpg")
    image_array = convert_image_to_array("/tmp/test.jpg")
    imagelist.append(image_array)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    print(np_image.shape)
    prediction = model.predict(np_image.reshape(1,256, 256, 3))
    leaf_class_category_mappings = {0:"Apple Apple scab", 1:"Apple Black rot", 2:"Apple Cedar apple rust",3:"Apple healthy", 4:"Background without leaves", 5:"Blueberry healthy", 6:"Cherry Powdery mildew", 7:"Cherry healthy",8:"Corn Cercospora leaf spotGray leaf spot", 9:"Corn Common rust", 10:"Corn Northern Leaf Blight", 11:"Corn healthy",12:"Grape Black rot", 13:"Grape Esca (Black Measles)", 14:"Grape Leaf blight (Isariopsis Leaf Spot)", 15:"Grape healthy",16:"Orange Haunglongbing (Citrus greening)", 17:"Peach Bacterial spot", 18:"Peach healthy",19:"Pepper  bell Bacterial spot",20:"Pepper  bell healthy", 21:"Potato Early blight", 22:"Potato Late blight", 23:"Potato healthy", 24:"Raspberry healthy",25:"Soybean healthy", 26:"Squash Powdery mildew", 27:"Strawberry Leaf scorch", 28:"Strawberry healthy", 29:"Tomato Bacterial spot",30:"Tomato Early blight", 31:"Tomato Late blight", 32:"Tomato Leaf Mold", 33:"Tomato Septoria leaf spot", 34:"Tomato Spider mitesTwo-spotted spider mite",35:"Tomato Target Spot", 36:"Tomato Tomato Yellow Leaf Curl Virus", 37:"Tomato Tomato mosaic virus", 38:"Tomato healthy"}
    print(prediction)
    #return ["leaf_class_category_mappings[np.argmax(prediction)]"
    #return [leaf_class_category_mappings[np.argmax(prediction)], "none"]
    predicteddisease = leaf_class_category_mappings[np.argmax(prediction)]
    reply = "Nothing to display"

    if predicteddisease.find("healthy") >= 0 :
        return [predicteddisease, "As the plant is healthy, nothing to worry about"]

    if predicteddisease.find("Background without leaves") >= 0 :
        gr.Error("Please upload a proper image")
        return ["Nothing to display", "None"]

    try :
        key1="sk"
        key2="-ico1dAjc3rZL3"
        key3="ssVVc4LT3BlbkFJwKTdY4IeCkMtZehqRpSU"

        openai.api_key = key1+key2+key3
        message = "What is "+predicteddisease+" and how to treat the disease"

        if message:
            messages = []
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
    except Exception as ex :
        print(ex)


    imageData = imageToProcess
    return [predicteddisease, reply]

def submitFeedback(correctOrWrong, plantName, userData):
    global dataToSend

    print(correctOrWrong)


    if correctOrWrong == "Not Satisfied" :
        dataToSend["PlantName"] = plantName
        dataToSend["Comments"] = userData
        dataToSend["FileContent"] = json.dumps(np.asarray(imageData).tolist())
        r = uploadFile()

        if r != None :
            res = json.loads(r.text)
            gr.Warning("Data Submitted for learning :" + res["Status"])
        else :
            gr.Error("Failed to upload the file for learning")

    saveStats(correctOrWrong)

with gr.Blocks() as gradioapp :

    gr.Markdown(
    """
        # AI based plant Disease Detection Application

    """
    )
    myControls["ImageInput"] = gr.Image(type="pil")

    controls = []

    myControls["ResultControl"] = gr.Textbox(label='Possible Disease could be ')
    myControls["AdditionalInfo"] = gr.TextArea(label='Additional Info')
    controls.append(myControls["ResultControl"])
    controls.append(myControls["AdditionalInfo"])


    predictBtn = gr.Button(value='Predict')
    predictBtn.click(predict, inputs=[myControls["ImageInput"]], outputs=controls)


    gr.Markdown()

    myControls["PredictionSelection"] = gr.Radio(["Satisfied", "Not Satisfied"], label="Feedback", info="Are you satisfied with the prediction?")
    #myControls["Feedback"] = gr.Checkbox(label="Is prediction wrong? If so, please provide the proper classification")
    myControls["PlantName"] = gr.Textbox(label='Specify the name of the plant')
    myControls["UserInput"] = gr.Textbox(label='What is the correct classification?')
    feedbackBtn = gr.Button(value='Submit Feedback')
    feedbackBtn.click(submitFeedback, inputs =[myControls["PredictionSelection"], myControls["PlantName"], myControls["UserInput"]])

    #app.queue().launch()
    gradioapp = gr.mount_gradio_app(app, gradioapp, path="/")

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)   # Ref: https://www.gradio.app/docs/interface
