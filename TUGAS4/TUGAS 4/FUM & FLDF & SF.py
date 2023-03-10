import cv2
import numpy as np

# memuat gambar
img = cv2.imread('Meong.jpg', 0)

# menerapkan filter masking unsharp
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
img_unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

# menerapkan filter Laplacian di domain frekuensi
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacian_fft = np.fft.fft2(laplacian, s=img.shape)
laplacian_fft_shifted = np.fft.fftshift(laplacian_fft)
fshift_filtered = fshift * laplacian_fft_shifted
f_filtered = np.fft.ifftshift(fshift_filtered)
img_laplacian = np.abs(np.fft.ifft2(f_filtered))

# Menerapkan pemfilteran selektif
f_low = np.zeros_like(fshift)
f_high = np.ones_like(fshift)
d0_low = 50  # Jarak cutoff untuk frekuensi rendah
d0_high = 10  # Jarak cutoff untuk frekuensi tinggi
u, v = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
d = np.sqrt((u - img.shape[1] // 2) ** 2 + (v - img.shape[0] // 2) ** 2)  # Jarak dari setiap titik ke pusat
f_low[d <= d0_low] = 1
f_high[d >= d0_high] = 0
f_combined = f_low + f_high
fshift_filtered = fshift * f_combined
f_filtered = np.fft.ifftshift(fshift_filtered)
img_selective = np.abs(np.fft.ifft2(f_filtered))

# menampilkan hasil
cv2.imshow('Original Image', img)
cv2.imshow('Unsharp Masked Image', img_unsharp)
cv2.imshow('Laplacian Filtered Image', img_laplacian)
cv2.imshow('Selective Filtered Image', img_selective)
cv2.waitKey(0)
cv2.destroyAllWindows()
