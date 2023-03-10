import cv2
import numpy as np

def median_filter(img, kernel_size):
    # Membuat gambar output kosong
    output = np.zeros_like(img)

    # Dapatkan radius kernel
    k = (kernel_size - 1) // 2

    # Iterasi di atas setiap piksel dalam gambar
    for y in range(k, img.shape[0] - k):
        for x in range(k, img.shape[1] - k):
            neighborhood = img[y - k:y + k + 1, x - k:x + k + 1]
            median = np.median(neighborhood)
            output[y, x] = median
    return output

# Membaca gambar input
img = cv2.imread("Cat.jpeg", cv2.IMREAD_GRAYSCALE)

# menerapkan filter median ke gambar
filtered_img = median_filter(img, kernel_size=3)

# menampilkan hasil
cv2.imshow("Input Image", img)
cv2.imshow("Filtered Image", filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()