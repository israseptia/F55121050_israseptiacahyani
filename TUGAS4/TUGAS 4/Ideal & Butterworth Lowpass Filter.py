import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca gambar
img = cv2.imread('Meong.jpg', 0)

# menghitung DFT dari gambar
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# menggeser nol frekuensi ke pusat
dft_shift = np.fft.fftshift(dft)

# membuat Ideal Lowpass Filter
rows, cols = img.shape
crow, ccol = rows//2, cols//2
radius = 30
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), radius, (1, 1), -1)

# membuat Butterworth Lowpass Filter
n = 2
d0 = 30
butterworth_lp = np.zeros((rows, cols, 2), np.uint8)
for i in range(rows):
    for j in range(cols):
        d = np.sqrt((i-crow)**2 + (j-ccol)**2)
        butterworth_lp[i,j] = 1 / (1 + (d/d0)**(2*n))

# mengaplikasikan filter ke gambar di domain frekuensi
dft_shift_filtered_ideal = dft_shift * mask
dft_shift_filtered_butterworth = dft_shift * butterworth_lp

# menggeser nol frekuensi kembali ke sudut kiri atas
dft_filtered_ideal = np.fft.ifftshift(dft_shift_filtered_ideal)
dft_filtered_butterworth = np.fft.ifftshift(dft_shift_filtered_butterworth)

# melakukan invers transformasi Fourier
img_back_ideal = cv2.idft(dft_filtered_ideal)
img_back_ideal = cv2.magnitude(img_back_ideal[:,:,0], img_back_ideal[:,:,1])

img_back_butterworth = cv2.idft(dft_filtered_butterworth)
img_back_butterworth = cv2.magnitude(img_back_butterworth[:,:,0], img_back_butterworth[:,:,1])

# menampilkan hasil
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img_back_ideal, cmap = 'gray')
plt.title('Hasil Ideal LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back_butterworth, cmap = 'gray')
plt.title('Hasil Butterworth LPF'), plt.xticks([]), plt.yticks([])
plt.show()
