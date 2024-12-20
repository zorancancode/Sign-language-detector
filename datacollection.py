import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize Hand Detector
detector = HandDetector(maxHands=1)

# Parameters
offset = 20
imgSize = 300
counter = 0
folder = "D:\\SignLanguageDetector\\Data\\Fuck You"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Adjust cropping boundaries to avoid out-of-bounds errors
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        if y2 > y1 and x2 > x1:  # Ensure the cropping region is valid
            imgCrop = img[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                # Height is greater than width
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                # Width is greater than height
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display cropped and resized images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        else:
            print("Invalid cropping dimensions.")

    # Display original image with hand detection
    cv2.imshow("Image", img)

    # Save image on key press
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")

    # Exit on pressing 'q'
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
