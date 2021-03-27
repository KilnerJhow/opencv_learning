#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt


#%%
#Carregando uma imagem
PATH = "imagem2.png"

flagReader= {  'Colorida': [1, cv2.IMREAD_COLOR],
                'Com transparencia': [-1, cv2.IMREAD_UNCHANGED],
                'Escala de cinza': [0, cv2.IMREAD_GRAYSCALE] 
            }

imgColor = cv2.imread(PATH)
imgTransparency = cv2.imread(PATH, flagReader['Com transparencia'][1])
imgGrayScale = cv2.imread(PATH, flagReader['Escala de cinza'][1])

print(imgColor.shape, imgTransparency.shape, imgGrayScale.shape)
# %%
# Visualizar uma imagem

windowName = "Hello, OpenCV"
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.imshow(windowName, imgColor)
cv2.waitKey()

# %%
