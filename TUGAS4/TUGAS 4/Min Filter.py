import cv2
import numpy as np

# memuat gambar
img = cv2.imread('Meong.jpg')

# mengatur ukuran kernel
ksize = 3

# membuat kernel
kernel = np.ones((ksize, ksize), np.uint8)

# menerapkan erosi
erosion = cv2.erode(img, kernel, iterations = 1)

# Tampilkan gambar asli dan difilter
cv2.imshow('Original', img)
cv2.imshow('Min Filter', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
