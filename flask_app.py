import json
import io
import os
import sys
from flask import *

from distutils.log import debug
from fileinput import filename
from flask import *
import numpy as np
from PIL import Image as im
import prometheus_client as prom
from fastapi import FastAPI, Request, Response

app = Flask(__name__)

imagePath = "/home/ubuntu/capstone_project/plant_leave_diseases_model/datasets/data/TRAIN"
success_counter = prom.Counter("prediction_success_counter", "Successful Prediction Counter")
failure_counter = prom.Counter("prediction_failure_counter", "Failure Prediction Counter")

def saveMetadata(filename):
    fileList = []
    if os.path.isfile("classification.meta") :
        f = open("classification.meta", "r")
        fileList = json.loads(f.read())
        f.close()

    try :
        idx = fileList.index(filename)
        if idx >= 0 :
            return
    except :
        fileList.append(filename)

    f = open("classification.meta", "w")
    f.write(json.dumps(fileList))
    f.close()


def saveImage(image, plantName, classificationInfo) :
    imageDir =  imagePath+os.sep+plantName+os.sep+classificationInfo
    if os.path.exists(imageDir) == False:

        os.makedirs(imageDir)

    filename = plantName+"___"+classificationInfo+".jpg"
    image.save(imageDir+os.sep+filename)
    saveMetadata(filename)

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/todb', methods = ['POST'])
def success():
    if request.method == 'POST':

        reqData = json.loads(request.data)
        fileContent = reqData["FileContent"]
        plantName = reqData["PlantName"]
        classificationInfo = reqData["Comments"].replace(" ", "_")
        deserializedContent = json.loads(fileContent) # we get a list
        npArray = np.array(deserializedContent) # now we get the ndarray
        image = im.fromarray((npArray * 1).astype(np.ubyte)) # now convert to image from ndarray

        saveImage(image, plantName, classificationInfo)


        res = {
            'Status':'Stored data for training'
        }

        return  make_response(json.dumps(res))

def loadStats():
    try :
        f = open("prediction.stats", "r")
        data = json.loads(f.read())
        f.close()
        return data
    except :
        return []

def saveStatsToFile(stats):
    f = open("prediction.stats", "w")
    f.write(json.dumps(stats))
    f.close()


@app.route('/predictionstats/get', methods=['POST','GET'])
def shareStats():
    stats = loadStats()
    r = {
        'Stats': stats
    }
    return make_response(json.dumps(stats))


@app.route('/predictionstats/get/prem', methods=['POST','GET'])
def shareStatsPrem():
    stats = loadStats()
    for r in stats:
        if int(r["PredictionStatus"]) == 1:
            success_counter.inc(1)
        else:
            failure_counter.inc(1)

    response = make_response(prom.generate_latest())
    response.headers["Content-Type"] = "text/plain"

    return response


@app.route('/predictionstats', methods=['POST'])
def saveStats():
    if request.method == 'POST' :
        req = json.loads(request.data)

        stats = loadStats()
        stats.append(req)
        saveStatsToFile(stats)

        print("Prediction Stats", req)

    return make_response("Stats Saved")
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5005)
