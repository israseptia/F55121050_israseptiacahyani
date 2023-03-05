import cv2
import numpy as np
from matplotlib import pyplot as plt

# memuat gambar skala abu-abu
img = cv2.imread('gambar.jpg', 0)

# melakukan pemerataan histogram
equ = cv2.equalizeHist(img)

# menampilkan gambar asli dan disamakan
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(equ, cmap='gray')
plt.title('Equalized Image'), plt.xticks([]), plt.yticks([])
plt.show()
