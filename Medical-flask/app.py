import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Some utilites
import numpy as np
from util import base64_to_pil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import random
import colorsys
from skimage.measure import find_contours
import base64
import cv2
import json
import time
from datetime import timedelta



#automatic serialization
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)



# Declare a flask app
app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)



#import mrcnn library
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

# define 81 classes that the coco model knowns about
class_names = ['BG', 'fracture']
# define the test configuration
class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
model.load_weights('models/mask_rcnn_bone_0030.h5', by_name=True)
print('Model loaded. Check http://127.0.0.1:5000/')






#routing for Res.html
@app.route('/Res')
def Res():
    return render_template('Res.html',val1=time.time())

#routing for Index.html
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#routing for predicting
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        r, g, b, a = img.split()
        img = Image.merge("RGB", (r, g, b))
        print(type(img))
        img = img_to_array(img)
        print(img.shape)
        result = model.detect([img], verbose=0)
        print(result)
        r=result[0]
        pred_proba = "{:.3f}".format(np.amax(r['scores']))
        print(pred_proba)
        image = my_display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        cv2.imwrite('static/processed.jpg',image)
        image = json.dumps(image, cls=MyEncoder)
        return jsonify(result=image, probability=pred_proba)

    return None

#mask the images
def my_display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('No instances to display')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    colors = random_colors(n_instances)
    height, width = image.shape[:2]
    
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{}{:.2f}'.format(label, score) if score else label
        image = cv2.putText(
                            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
                            )
    return image

#utils functions
def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
