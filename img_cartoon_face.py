# plot photo with detected faces using opencv cascade classifier
from cv2 import cv2
from cv2.cv2 import imread, inpaint
from cv2.cv2 import imshow
from cv2.cv2 import waitKey
from cv2.cv2 import destroyAllWindows
from cv2.cv2 import CascadeClassifier
from cv2.cv2 import rectangle
import numpy as np
# load the photograph
pixels = imread('test1.png')
pixels_orig = pixels
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
# perform face detectionq
bboxes = classifier.detectMultiScale(pixels)
x, y, width, height = bboxes[0]
face_image = pixels_orig[y:y+height,x:x+width]
imshow('original face', face_image)
ksize = (30, 30)
imageAllBlurred = cv2.blur(face_image,ksize=ksize) # with a kernel size value 'ksize' equals to 3 for example
for i in range(10):
    imageAllBlurred = cv2.blur(face_image,ksize=ksize)

imageAllBlurred = np.where(imageAllBlurred<face_image,imageAllBlurred,face_image)
norm_image = cv2.normalize(imageAllBlurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# show the image
imshow('cartoon image blurred', norm_image)

waitKey(0)
# close the window
destroyAllWindows()