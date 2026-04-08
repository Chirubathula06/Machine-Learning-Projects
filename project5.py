# Face Detection using OpenCV (FINAL FIX)

import cv2
import matplotlib.pyplot as plt

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load Image
img = cv2.imread("githubphoto.jpeg")

# Check image
if img is None:
    print("Error: Image not found")
    exit()

# Convert to Gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

# Draw Rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Convert BGR → RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show using matplotlib (NO ERROR)
plt.imshow(img_rgb)
plt.title("Face Detection")
plt.axis("off")
plt.show()
