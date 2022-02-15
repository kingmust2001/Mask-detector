# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 11:47:37 2021

@author: user
"""
from flask import Flask
from flask import render_template, jsonify
import os
from flask import request
from werkzeug.utils import secure_filename
import shutil


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def Mask_Detector(img):
    "build up network, import cfg and yolo weight"
    net = cv2.dnn.readNetFromDarknet(r"./yolov3.cfg", r"./yolov3_1100.weights")
    "output layer for 3 different size map"
    layerName = net.getLayerNames()
    output_Layer = [layerName[i[0]-1] for i in net.getUnconnectedOutLayers()]
    "load related param"
    classes = [line.strip() for line in open(r"./obj.names")]
    colors = [(0,0,255), (255,0,0), (0,255,0)]
    """applied yolo model detect
    320 x 320 (high speed, less accuracy)
    416 x 416 (moderate speed, moderate accuracy)
    608 x 608 (less speed, high accuracy)"""
    img = cv2.resize(img, None, fx=0.4, fy=0.4) 
    height, width, channel = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob) 
    outs = net.forward(output_Layer)
    "get bndbox information"
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            #print(detection) #[9.8863834e-01 9.9294084e-01 5.7538722e-02 1.4175595e-01 1.3973890e-06 0.0000000e+00 0.0000000e+00 0.0000000e+00]
            tx, ty, tw, th, confidence = detection[0:5]
            scores = detection[5:]
            class_id = np.argmax(scores) #取出值最大並返回index
            if confidence > 0.3:   
                center_x = int(tx * width)
                center_y = int(ty * height)
                w = int(tw * width)
                h = int(th * height)
                
                # 取得箱子方框座標
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    "NMS - non-maximum suppression applied"
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4) 
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-5), font, 1, color, 1)
    plt.rcParams['figure.figsize'] = [21,16]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

try:
    shutil.rmtree('./static')
    os.mkdir('./static')
except:
    pass
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/", methods = ["POST", "GET"])
def homePage():
    return render_template("mask_detector.html")
@app.route("/mask_detector_result", methods = ["POST", "GET"])
def maskDetect():
    try:
        os.remove("./static/detection.jpg")
    except:
        pass
    if request.method == 'POST':
        "get img from files"
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "img type：png、PNG、jpg、JPG、bmp"})
        f.save("./static/{}".format(f.filename))
        # basepath = os.path.dirname(__file__)
        # upload_path = os.path.join(basepath, 'static', secure_filename(f.filename)) 
        # img = cv2.imread("./static/{}".format(f.filename))
        img = cv2.imread("./static/{}".format(f.filename))
        detection = Mask_Detector(img)
        detection = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./static/{}".format("detection_"+f.filename), detection)
        return render_template("mask_detector_result.html", data = f.filename, detect = "detection_"+f.filename)

if __name__ == 'main':
    app.run() 
# app.run()       