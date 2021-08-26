# Streamlit-based Camera Object Detection Data App

A data app for detecting objects in any given picture. Either upload your own picture for detection or enter a URL referecing a web cam still image or any other Web picture.

## Pretrained model attributation

The pretrained binary model (binary file with extension .pb and .pbtxt) was downloaded and originates from [Model Zoo](https://modelzoo.co/) referenced by [OpenCV](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API). Those files contain both topology and weights of the trained network. You may download one of them from Model Zoo, in example ssd_mobilenet_v1_coco (MobileNet-SSD trained on COCO dataset).

## Picture attributation

The example picture under /src/cam_image.jpg was taken from [Pixabay](https://pixabay.com/de/photos/porsche-stra%c3%9fe-retro-wagen-5665390/) under the free [Pixabay License](https://pixabay.com/de/service/license/).
