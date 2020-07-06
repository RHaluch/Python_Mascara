'''
Parte do codigo foi gentilmente fornecido através de email pelo pesquisador Adrian Rosebrock da comunidade PyImageSearch
https://www.pyimagesearch.com/

Alterações feitas para incluir o servidor Flask para receber imagens do app e retornar imagem com resultado
'''
# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import socket
import flask
from flask import send_file
import matplotlib.pyplot as plt

app = flask.Flask(__name__)

# load our serialized face detector model from disk
path = os.getcwd()
print("[INFO] loading face detector model...")
prototxtPath = path + "\\deploy.prototxt"
weightsPath = path + "\\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(path + "\\mask_detector.model")

@app.route('/', methods=['GET', 'POST'])
def handle_request():
    imagefile = flask.request.files['image']
    imagefile.save("recebida.jpg")
    print("Imagem Recebida!")

    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread("recebida.jpg")
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mascara" if mask > withoutMask else "Sem Mascara"
            color = (0, 255, 0) if label == "Mascara" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)

    print("Pronto")
    #show the output image
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)
    cv2.imencode('.jpg',image)
    cv2.imwrite('resultado.jpg',image)
    #if 'label' not in locals():
    #    return 'Face não encontrada!'
    #else:
    return send_file('resultado.jpg', mimetype='image/jpg', cache_timeout=0)

app.run(host="0.0.0.0", port=5000, debug=True)