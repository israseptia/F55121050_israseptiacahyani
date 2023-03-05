import cv2

# Load gambar pertama dan kedua
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Ubah kedua gambar menjadi grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Hitung perbedaan pixel antara kedua gambar
diff = cv2.absdiff(gray_img1, gray_img2)

# Thresholding pada citra perbedaan untuk menghilangkan noise
thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]

# Mencari kontur pada citra perbedaan
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Gambar kotak pada kontur yang terdeteksi pada citra asli
for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Tampilkan gambar asli dan gambar perbandingan
cv2.imshow("Original 1", img1)
cv2.imshow("Original 2", img2)
cv2.imshow("Difference", diff)
cv2.imshow("Thresholded Difference", thresh)

# Tunggu input dari pengguna
cv2.waitKey(0)
cv2.destroyAllWindows()
