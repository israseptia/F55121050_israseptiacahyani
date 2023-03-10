import cv2
import numpy as np

# memuat gambar
img = cv2.imread('Meong.jpg', 0)

# menerapkan filter lowpass Gaussian
kernel_size = 15
sigma = 5
kernel = cv2.getGaussianKernel(kernel_size, sigma)
kernel = np.outer(kernel, kernel.transpose())
kernel = kernel / kernel.sum()  # Normalize the kernel
filtered_img = cv2.filter2D(img, -1, kernel)

# menerapkan filter highpass yang ideal
d = 50
h, w = img.shape
cx, cy = w // 2, h // 2
mask = np.zeros((h, w), np.uint8)
cv2.circle(mask, (cx, cy), d, 255, -1)
mask = 1 - mask / 255
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fshift_filtered = fshift * mask
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered = np.fft.ifft2(f_filtered)
img_filtered = np.abs(img_filtered)

# menampilkan hasil
cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Lowpass Filtered Image', filtered_img)
cv2.imshow('Ideal Highpass Filtered Image', img_filtered.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
