import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab
img1 = plt.imread('test/Camilo/rostro_12.jpg')
img2 = plt.imread('test/Camilo/rostro_18.jpg')
img3 = plt.imread('test/Camilo/rostro_19.jpg')
img4 = plt.imread('test/Camilo/rostro_42.jpg')
img5 = plt.imread('test/Camilo/rostro_1379.jpg')
img6 = plt.imread('test/Camilo/rostro_7.jpg')

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('rostro_12')
plt.subplot(232),plt.imshow(img2,'gray'),plt.title('rostro_18')
plt.subplot(233),plt.imshow(img3,'gray'),plt.title('rostro_19')
plt.subplot(234),plt.imshow(img4,'gray'),plt.title('rostro_42')
plt.subplot(235),plt.imshow(img5,'gray'),plt.title('rostro_1379')
plt.subplot(236),plt.imshow(img6,'gray'),plt.title('rostro_7')
plt.show()

plt.savefig()