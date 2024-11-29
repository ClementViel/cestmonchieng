import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import numpy as np
import imutils

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]

model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
)

img_path = '/Users/clem/Downloads/PXL_20240406_130454992.jpg'
img = keras.utils.load_img(img_path)
x = np.array(img)

inference_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True
)

preds = model.predict(inference_resizing([x]))

(startX, startY, endX, endY) = preds["boxes"][0][0]
(startx, starty, endx, endy) = preds["boxes"][0][1]
# (startX, startY, endX, endY) = preds["boxes"][0][1]
print(preds["boxes"][0][0])
print(preds["boxes"][0][1])
print(preds["boxes"][0][2])
print(class_ids[preds["classes"][0][0]])
print(class_ids[preds["classes"][0][1]])
print(class_ids[preds["classes"][0][2]])
# load the input image (in OpenCV format), resize it such that it
# fits on our screen, and grab its dimensions
image = cv2.imread(img_path)
image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
(h, w) = image.shape[:2]
# scale the predicted bounding box coordinates based on the image
# dimensions
print("dimension image : ", h, w)
startX = int(startX)
startY = int(startY)
endX = int(endX)
endY = int(endY)
# draw the predicted bounding box on the image
# crop the dog !!!!
for idx in range(0, len(class_ids)):
    if class_ids[preds["classes"][0][idx]] == "Dog":
        print("There is a dog")
        print(preds["boxes"][0][idx])
        (startX, startY, width, height) = preds["boxes"][0][idx]
        cv2.rectangle(image, (int(startX), int(startY)), (int(startX)+int(width), int(startY) + int(height)),
                      (0, 255, 0), 2)
        cropped_image = image[int(startY):int(
            startY)+int(height), int(startX):int(startX)+int(width)]
        cv2.imshow("crop", image)
        cv2.waitKey(0)
        filename = "/Users/clem//Projets/prog/cestmonchieng/cropped" + \
            str(idx) + ".jpg"
        cv2.imwrite(filename, cropped_image)
# show the output image
