import numpy as np
import cv2

# membaca gambar
img = cv2.imread('gambar.jpg')

# menentukan ukuran kernel
ksize = 7

# membuat kernel filter rata-rata
kernel = np.ones((ksize, ksize), np.float32) / (ksize**2)

# melakukan konvolusi pada gambar menggunakan kernel filter rata-rata
filtered_img = cv2.filter2D(img, -1, kernel)

# menampilkan gambar asli dan hasil filter
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

