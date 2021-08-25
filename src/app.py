import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2 
import requests

DEFAULT_CONFIDENCE_THRESHOLD = 0.70
confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

img = cv2.imread('/app/src/cam_image.jpg')

def download_cam_image(camera_url):
    response = requests.get(camera_url)
    file = open("/app/src/cam_image.jpg", "wb")
    file.write(response.content)
    file.close()

@st.cache
def class_labels():
    classLabels = []
    with open('/app/src/labels.txt', 'rt') as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')
    return classLabels

def classify(img, confidence_threshold):
    results = []
    config_file = '/app/src/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = '/app/src/frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0/127.5)
    model.setInputMean((127.5,127.5,127.5))
    model.setInputSwapRB(True)
    model.setInputCrop(False)
    img = cv2.imread('/app/src/cam_image.jpg')
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.2)
    if len(ClassIndex) > 1:
        # draw detection boxes and labels
        font = cv2.FONT_HERSHEY_PLAIN
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if conf > confidence_threshold:
                # append the object entry to our result table
                results.append([class_labels()[ClassInd-1], conf])
                cv2.rectangle(img, boxes, (255, 0, 0), 2)
                cv2.putText(img, 
                            class_labels()[ClassInd-1], 
                            (boxes[0]+10, boxes[1]+40), 
                            font, 
                            fontScale=1.0, 
                            color=(0,255,0))
    return results, img

results = []

st.title('Camera Object Detection')

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if img_file_buffer is not None:
    with Image.open(img_file_buffer) as im:
        rgb_im = im.convert('RGB')
        rgb_im.save('/app/src/cam_image.jpg')
        results = classify(img, confidence_threshold)

confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

results, img = classify(img, confidence_threshold)
st.image(img)

df = pd.DataFrame(results, columns=(['Object', 'Confidence']))
st.dataframe(df)  