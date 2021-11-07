# pydscodes
## Python Data Science Codes

1. Rectangle Optimization using Empty Area Detection Problem - rectangular_optimization.ipynb
2. Cricket Stats - Simple Scorecard - Read MongoDB JSON and calculate batting aggregate metrics
3. Util Functions - Functions for MongoDB retrieval and update/create
4. NLP SKLearn Custom Transformations - Notebook with custom transformers for pandas dataframe based processing on more than one columnar unstructured text
5. Image Processing - Notebook with ability to read the binary RGB values of images and perform CNN classification of the digits identification

### Image Processing Task 2 - Create a Cartoon Image of Potrait Image using OPENCV-PYTHON

The code uses python-cv2, numpy, and matplotlib packages to perform the image processing function.

```
pip install cv2
pip install numpy
pip install matplotlib
```
Let's import the required functions:

* imread - to read images
* imshow - to display the images
* CascadeClassifier - for face detection from a potrait image
* numpy - for array computation
* matplotlib.pyplot - for plotting the images
```
from cv2 import cv2
from cv2.cv2 import imread
from cv2.cv2 import imshow
from cv2.cv2 import CascadeClassifier
import numpy as np
import matplotlib.pyplot as plt
```
Specify plotting the images in the notebook.
```
%matplotlib inline
```
Load the original image
```
# load the photograph
pixels = imread('face_cartoon_orig.png')
pixels_orig = np.flip(pixels, axis=-1)
plt.imshow(pixels_orig)
plt.show()
```
![Original Face Potrait Image](https://github.com/anands85/pydscodes/blob/main/face_cartoon_orig.png)

```
# load the pre-trained model
classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform face detectionq
bboxes = classifier.detectMultiScale(pixels)
x, y, width, height = bboxes[0]
face_image = pixels_orig[y:y+height,x:x+width]
plt.imshow(face_image)
plt.show()
```
![Original Face Detected](https://github.com/anands85/pydscodes/blob/main/face_cartoon_orig_detect.png]

**ksize** is the size of the mask for blur function to perform on the original image. Blurring removes the differences in pixel values in the neighbor pixels.

As we blur the image, the pixels smoothens out and flatter image regions are obatined for the cartoon image creation. The contours due to image pixel differences are normalized with the median blur images.
```
ksize = 3
imageAllBlurred = cv2.medianBlur(face_image,ksize=ksize) # with a kernel size value 'ksize' equals to 3 for example
for i in range(2):
imageAllBlurred = cv2.medianBlur(face_image,ksize=ksize)
imageAllBlurred = np.where(imageAllBlurred<face_image,face_image,imageAllBlurred)
norm_image = cv2.normalize(imageAllBlurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# show the image of the cartoon face
plt.imshow(norm_image)
plt.show()
```
![Output Cartoon Face](https://github.com/anands85/pydscodes/blob/main/output.png]

```
# load the catoon photograph
toonpixels = imread('face_cartoon_toon.png')
toonpixels = np.flip(toonpixels, axis=-1) 
bboxes = classifier.detectMultiScale(toonpixels)
x, y, width, height = bboxes[0]
face_image = toonpixels[y:y+height,x:x+width]
plt.imshow(face_image)
plt.show()
```
![Original Images](https://github.com/anands85/pydscodes/blob/main/face_cartoon.png]

## License
Feel free to modify, change, extend, include the codes as a comercial or free product.
