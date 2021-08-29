import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2 
import requests
import os

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

def inject(tag):
    with open(os.path.dirname(st.__file__) + "/static/index.html", 'r') as file:
        str = file.read()
        if str.find(tag) == -1:
            idx = str.index('<head>')
            new_str = str[:idx] + tag + str[idx:]
            with open(os.path.dirname(st.__file__) + "/static/index.html", 'w') as file:
                file.write(new_str)

inject('<script type="text/javascript" src="https://js-cdn.dynatrace.com/jstag/148709fdc4b/bf74387hfy/f9077358a180e88d_complete.js" crossorigin="anonymous"></script>')    

@st.cache
def download_trained_model():
    # download and save model
    r = requests.get('https://github.com/wolfgangB33r/camera-object-detection/blob/main/model/graph.pb?raw=true')  
    with open('graph.pb', 'wb') as f:
        f.write(r.content)
    # download and save model config
    r = requests.get('https://github.com/wolfgangB33r/camera-object-detection/blob/main/model/model_config.pbtxt?raw=true')  
    with open('model_config.pbtxt', 'wb') as f:
        f.write(r.content)
    # download and save model lables file
    r = requests.get('https://github.com/wolfgangB33r/camera-object-detection/blob/main/model/labels.txt?raw=true')  
    with open('labels.txt', 'wb') as f:
        f.write(r.content)

download_trained_model()

def download_cam_image(picture_url):
    resp = requests.get(picture_url, stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

img = download_cam_image('https://github.com/wolfgangB33r/camera-object-detection/raw/main/images/cam_image.jpg')

@st.cache
def class_label(classIndex):
    classLabels = [] 
    with open('labels.txt', 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
    return classLabels[classIndex-1]

def classify(img, confidence_threshold):
    results = []
    config_file = 'model_config.pbtxt'
    frozen_model = 'graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)
    model.setInputCrop(False)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.2)
    if len(ClassIndex) > 1:
        # draw detection boxes and labels
        font = cv2.FONT_HERSHEY_PLAIN
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if conf > confidence_threshold:
                # append the object entry to our result table
                results.append([ClassInd, class_label(ClassInd), conf])
                cv2.rectangle(img, boxes, (255, 0, 0), 2)
                cv2.putText(img, 
                            class_label(ClassInd), 
                            (boxes[0]+10, boxes[1]+40), 
                            font, 
                            fontScale=1.0, 
                            color=(0,255,0))
    return results, img

results = []

st.title('Camera Object Detection')

st.markdown('Source at [GitHub](https://github.com/wolfgangB33r/camera-object-detection), read the companion [blog](https://wolfgangb33r.medium.com/how-to-uplevel-your-security-cam-by-detecting-objects-with-a-streamlit-data-app-aa2cdfef391d).')

url = st.text_input('Specify a Picture URL')
if url is not None and url.startswith('http'):
    img = download_cam_image(url)
    img_file_buffer = None

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.asarray(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

results, img = classify(img, confidence_threshold)
st.image(img)

df = pd.DataFrame(results, columns=(['Class', 'Label', 'Confidence']))
st.dataframe(df)  