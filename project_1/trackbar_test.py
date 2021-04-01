#inicio
import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_NAME = 'Projeto 1'
PATH_IMG_1 = "image.jpg"
PATH_IMG_2 = "image2.jpg"

def show_image(img):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey()

def nothing(x):
    pass

# imgRead = cv2.imread(PATH_IMG_1, cv2.IMREAD_GRAYSCALE)

imgRead = cv2.imread(PATH_IMG_2, cv2.IMREAD_GRAYSCALE)

# print(imgRead_1.shape, imgRead_2.shape)

figPlot = plt.figure()

# Limiarizando imagem 1
theta = 30
# create trackbars for theta change
cv2.namedWindow(WINDOW_NAME)
cv2.createTrackbar('Theta',WINDOW_NAME,0,255, nothing)

while(1):

    valor, img1Binary = cv2.threshold(imgRead, theta, 255, cv2.THRESH_BINARY)

    ROI = cv2.bitwise_and(imgRead, imgRead, mask=img1Binary)

    histImg1_2 = histImg_1 = cv2.calcHist([imgRead],[0],img1Binary,[256],[0,256])

    cv2.imshow(WINDOW_NAME, ROI)

    # cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    theta = cv2.getTrackbarPos('Theta',WINDOW_NAME)


cv2.destroyAllWindows()
