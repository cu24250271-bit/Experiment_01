# Experiment_01
import cv2
import numpy as np
import matplotlib.pyplot as plt



img = cv2.imread("image.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

print("Original image shape:", img.shape)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


resized = cv2.resize(img, (400, 300))



scaled = cv2.resize(img, None, fx=0.5, fy=0.5)


negative = 255 - img



x1, y1 = 100, 100
x2, y2 = 400, 400

roi = img[y1:y2, x1:x2]



blurred = cv2.GaussianBlur(img, (5, 5), 0)



edges = cv2.Canny(gray, 100, 200)



_, threshold = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)


plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.imshow(rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(hsv)
plt.title("HSV")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(lab)
plt.title("LAB")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(threshold, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.tight_layout()
plt.show()


print("\n===== ROI ANALYSIS =====")
print("ROI shape:", roi.shape)
print("ROI mean:", roi.mean())
print("ROI minimum:", roi.min())
print("ROI maximum:", roi.max())



height, width, channels = img.shape

print("\n===== IMAGE INFORMATION =====")
print("Width:", width)
print("Height:", height)
print("Channels:", channels)
print("Total pixels:", width * height)
print("Data type:", img.dtype)