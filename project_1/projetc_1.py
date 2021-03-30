
#%%
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


imgRead_1 = cv2.imread(PATH_IMG_1, cv2.IMREAD_GRAYSCALE)

imgRead_2 = cv2.imread(PATH_IMG_2, cv2.IMREAD_GRAYSCALE)

print(imgRead_1.shape, imgRead_2.shape)

figPlot = plt.figure()



#%%
# Fazendo a filtragem da imagem 1

blur = cv2.blur(imgRead_1, (5,5))

gaussianBlur = cv2.GaussianBlur(imgRead_1, (11,11), 5)

medianBlur = cv2.medianBlur(imgRead_1,5)

laplacian = cv2.Laplacian(imgRead_1, cv2.CV_64F)

#Plotando as imagens

#subplot(linha, coluna, index)
#index vai de 1 até o número máximo de items, dado por linha x coluna

figPlot.add_subplot(3,3,1)
plt.imshow(imgRead_1, cmap='gray')
plt.title("Original")

figPlot.add_subplot(3,3,2)
plt.imshow(blur, cmap='gray')
plt.title("Blur - Image 1")

figPlot.add_subplot(3,3,3)
plt.imshow(gaussianBlur, cmap='gray')
plt.title("Gaussian Blur - Image 1")

figPlot.add_subplot(3,3,4)
plt.imshow(medianBlur, cmap='gray')
plt.title("Median Blur - Image 1")

plt.show()



#%%
#plotando histograma da imagem 1
histImg_1 = cv2.calcHist([imgRead_1],[0],None,[256],[0,256])
histImg_equ = cv2.equalizeHist(imgRead_1)
after_histImg_1 = cv2.calcHist([histImg_equ], [0], None, [256], [0,256])

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) #Histograma do tipo clahe
histImg1Clahe = clahe.apply(imgRead_1)
afterhistImg1Clahe = cv2.calcHist([histImg1Clahe], [0], None, [256], [0,256])

plt.subplot(321), plt.imshow(imgRead_1, 'gray'), plt.tight_layout(pad=1.0)
plt.subplot(322), plt.plot(histImg_1), plt.tight_layout(pad=1.0), plt.xlim([0,256])
plt.subplot(323), plt.imshow(histImg_equ, 'gray'), plt.tight_layout(pad=1.0)
plt.subplot(324), plt.plot(after_histImg_1), plt.tight_layout(pad=1.0)#, plt.xlim([0,256])
plt.subplot(325), plt.imshow(histImg1Clahe), plt.tight_layout(pad=1.0)
plt.subplot(326), plt.plot(afterhistImg1Clahe), plt.tight_layout(pad=1.0)#, plt.xlim([0,256])

plt.show()

# show_image(histImg1Clahe)

#%%
# Limiarizando imagem 1


theta = 30

valor, img1Binary = cv2.threshold(imgRead_1, theta, 255, cv2.THRESH_BINARY)

# show_image(img1Binary)

ROI = cv2.bitwise_and(imgRead_1, imgRead_1, mask=img1Binary)

histImg1_2 = histImg_1 = cv2.calcHist([imgRead_1],[0],img1Binary,[256],[0,256])

edges = cv2.Canny(imgRead_1,100,200)

plt.subplot(2,2,1), plt.imshow(imgRead_1, 'gray')
plt.subplot(2,2,2), plt.imshow(ROI, 'gray')
plt.subplot(2,2,3), plt.plot(histImg1_2)
plt.subplot(2,2,4), plt.imshow(edges, 'gray')




#%%
#plotando histograma da imagem 2
histImg2 = cv2.calcHist([imgRead_2],[0],None,[256],[0,256])
histImgEqual = cv2.equalizeHist(imgRead_2)
after_histImg2 = cv2.calcHist([histImgEqual], [0], None, [256], [0,256])

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) #Histograma do tipo clahe
histImg2Clahe = clahe.apply(imgRead_2)
afterhistImg2Clahe = cv2.calcHist([histImg2Clahe], [0], None, [256], [0,256])

plt.subplot(321), plt.imshow(imgRead_2, 'gray'), plt.tight_layout(pad=1.0)
plt.subplot(322), plt.plot(histImg2), plt.tight_layout(pad=1.0), plt.xlim([0,256])
plt.subplot(323), plt.imshow(histImgEqual, 'gray'), plt.tight_layout(pad=1.0)
plt.subplot(324), plt.plot(after_histImg2), plt.tight_layout(pad=1.0)#, plt.xlim([0,256])
plt.subplot(325), plt.imshow(histImg2Clahe), plt.tight_layout(pad=1.0)
plt.subplot(326), plt.plot(afterhistImg2Clahe), plt.tight_layout(pad=1.0)#, plt.xlim([0,256])

plt.show()

show_image(histImg2Clahe)


#%%
#Erosão e dilatação da imagem 1

kernel = np.ones((5,5), np.uint8) #matriz 5x5 cheia de 1s

erosion = cv2.erode(imgRead_1, kernel, iterations=1)
dilate = cv2.dilate(imgRead_1, kernel, iterations=1)

opening = cv2.morphologyEx(imgRead_1, cv2.MORPH_OPEN, kernel) #erosão seguida de dilatação
closing = cv2.morphologyEx(imgRead_1, cv2.MORPH_CLOSE, kernel) #dilatação seguida de erosão

plt.subplot(231), plt.imshow(imgRead_1, 'gray'), plt.title("Original"), plt.tight_layout(pad=1.0)
plt.subplot(232), plt.imshow(erosion, 'gray'), plt.title("Erosao"), plt.tight_layout(pad=1.0)
plt.subplot(233), plt.imshow(dilate, 'gray'), plt.title("Dilatacao"), plt.tight_layout(pad=1.0)
plt.subplot(234), plt.imshow(opening, 'gray'), plt.title("Opening"), plt.tight_layout(pad=1.0)
plt.subplot(235), plt.imshow(closing, 'gray'), plt.title("Closing"), plt.tight_layout(pad=1.0)


plt.show


# %%
