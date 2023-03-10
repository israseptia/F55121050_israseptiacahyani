import cv2
import numpy as np

# Load image
img = cv2.imread('Meong.jpg')

# Set kernel size
ksize = 3

# Create kernel
kernel = np.ones((ksize, ksize), np.uint8)

# Apply dilation
dilation = cv2.dilate(img, kernel, iterations = 1)

# Show original and filtered image
cv2.imshow('Original', img)
cv2.imshow('Max Filter', dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
