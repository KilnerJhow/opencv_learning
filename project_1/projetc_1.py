#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

WINDOW_NAME = 'Projeto 1'
PATH_IMG_1 = "image.jpg"
PATH_IMG_2 = "image2.jpg"

imgRead_1 = cv2.imread(PATH_IMG_1, cv2.IMREAD_GRAYSCALE)

imgRead_2 = cv2.imread(PATH_IMG_2, cv2.IMREAD_GRAYSCALE)

print(imgRead_1.shape, imgRead_2.shape)

#%%
# Fazendo a filtragem da imagem 1

blur = cv2.blur(imgRead_1, (5,5))

gaussianBlur = cv2.GaussianBlur(imgRead_1, (11,11), 5)

medianBlur = cv2.medianBlur(imgRead_1,5)

#Plotando as imagens

#subplot(linha, coluna, index)
#index vai de 1 até o número máximo de items, dado por linha x coluna

figPlot = plt.figure()
figPlot.add_subplot(3,3,1)
plt.imshow(imgRead_1, cmap='gray')
plt.title("Image 1")

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
#plotando histograma
histImg_1 = cv2.calcHist([imgRead_2],[0],None,[256],[0,256])
histImg_equ = cv2.equalizeHist(imgRead_2)
after_histImg_1 = cv2.calcHist([histImg_equ], [0], None, [256], [0,256])

plt.subplot(231), plt.imshow(imgRead_2, 'gray')
plt.subplot(233), plt.plot(histImg_1)
plt.xlim([0,256])
plt.subplot(234), plt.imshow(histImg_equ, 'gray')
plt.subplot(236), plt.plot(after_histImg_1)
plt.xlim([0,256])

plt.show()

# cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
# cv2.imshow(WINDOW_NAME, imgColor_1)
# cv2.waitKey()

# %%
