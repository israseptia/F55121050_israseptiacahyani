import cv2
import numpy as np

# memuat gambar
img = cv2.imread('Meong.jpg', 0)

# menerapkan filter highpass Butterworth
d0 = 50  # Jarak cutoff dari pusat
n = 2  # Urutan filter
h, w = img.shape
cx, cy = w // 2, h // 2  # Pusat gambar
u, v = np.meshgrid(np.arange(w), np.arange(h))
d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)  # Jarak dari setiap titik ke pusat
bhpf = 1 / (1 + (d0 / d) ** (2 * n))  # filter Butterworth highpass
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fshift_filtered = fshift * bhpf
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered_butter = np.fft.ifft2(f_filtered)
img_filtered_butter = np.abs(img_filtered_butter)

# menerapkan filter highpass Gaussian
sigma = 5
ghpf = 1 - np.exp(-(d ** 2) / (2 * sigma ** 2))  # filter highpass Gaussian
fshift_filtered = fshift * ghpf
f_filtered = np.fft.ifftshift(fshift_filtered)
img_filtered_gaussian = np.fft.ifft2(f_filtered)
img_filtered_gaussian = np.abs(img_filtered_gaussian)

# manampilkan hasil
cv2.imshow('Original Image', img)
cv2.imshow('Butterworth Highpass Filtered Image', img_filtered_butter.astype(np.uint8))
cv2.imshow('Gaussian Highpass Filtered Image', img_filtered_gaussian.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
