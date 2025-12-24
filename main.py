import cv2
import numpy as np
import sys
import os
from functions import MorphologyOperations, SmoothingFilters
from shape_operations import shape_operations

figure = shape_operations()

print()
print("-"*20)
caminho_atual = os.path.dirname(__file__)

print(f"caminho --> {caminho_atual}")

caminho_imagem_exemplos_dados = os.path.join(caminho_atual, "inputs", "chosen")
caminho_imagem_exemplos_criados = os.path.join(caminho_atual, "inputs", "meus_inputs")
caminho_imagem_base_com_lipobag = os.path.join(caminho_atual, "inputs", "base_com_lipobag")
caminho_videos = os.path.join(caminho_atual,"inputs","videos")

print(f"caminho final até a imagem --> {caminho_imagem_exemplos_dados}")
print(f"caminho final até a imagem --> {caminho_imagem_exemplos_criados}")
print("-"*20)
print()

#=====================================================================#

def nothing(x):
    pass

#=====================================================================#

print("[ 0 ] --> DETECTAR USANDO A CAMERA")
print("[ 1 ] --> DETECTAR UM ARQUIVO DE IMAGEM")

#choice = int(input("ESCOLHA UMA DAS OPCOES: "))
choice = 0

if choice == 0:
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(caminho_videos+"/14.mp4")
    frame_time = 50
else:
    frame_time = 1000

#=====================================================================#

cv2.namedWindow("CONTROL", cv2.WINDOW_NORMAL)
cv2.namedWindow("IMAGEM-BINARIZADA / IMAGEM-CONTORNADA", cv2.WINDOW_NORMAL)

cv2.createTrackbar("Sm.FT", "CONTROL", 1, 4, nothing)
cv2.createTrackbar("Morph.FT", "CONTROL", 4, 7, nothing)
#cv2.createTrackbar("Matrix.Num", "CONTROL", 0, 50, nothing)
cv2.createTrackbar("Threshold", "CONTROL", 100, 255, nothing)
#cv2.createTrackbar("Color", "CONTROL", 0, 2, nothing)
#cv2.createTrackbar("Ar.Cross", "CONTROL", 2000, 100000, nothing)
#cv2.createTrackbar("Ar.Square", "CONTROL", 2000, 100000, nothing)
#cv2.createTrackbar("Ar.Circle", "CONTROL", 2000, 100000, nothing)

#=====================================================================#

print(f"a sua escolha foi de {choice}")

while (True):
    
    if choice == 0:
        ret, sorce_image = cap.read()

        if not ret:
            print("nao iniciou")
            break

#         if sorce_image is None:
#             sys.exit("ERROR: COULD NOT READ THE IMAGE")


    
    #criado um backup da foto de entrada original
    sorce_image_copy = sorce_image.copy()

    #hsv_sorce_image = cv2.cvtColor(sorce_image, cv2.COLOR_BGR2HSV)
    #-------------------------------------------------------------------------------------------------------#


    #-----------------------------------FUNCIONAMENTO DOS SLIDERS--------------------------------------------#
    sliders1 = cv2.getTrackbarPos("Sm.FT", "CONTROL")               # responsável por aplicar os filtros de suavização
    sliders2 = cv2.getTrackbarPos("Morph.FT", "CONTROL")            # responsável por aplicar os filtros de morfologia
    sliders5 = cv2.getTrackbarPos("Threshold", "CONTROL")           # responsável por aplicar o número do limiar na imagem

    #--------------------------------------------------------------------------------------------------------#

    #---------------------CONVERTENDO A IMAGEM DE ENTRADA PARA O TOM DE CINZA#---------------------#
    sorce_image_gray_image = cv2.cvtColor(sorce_image, cv2.COLOR_BGR2GRAY)
    
    #---------------------APLICANDO FILTROS DE SUAVIZAÇÃO-------------------------------#
    
    sorce_image_smoothing_filter = SmoothingFilters(sliders1, sorce_image_gray_image)                            
    
    #-------------------------------APLICANDO FILTROS MORFOLÓGICOS-------------------------------#

    sorce_image_morphology_operations = MorphologyOperations(sorce_image_smoothing_filter, sliders2)

    #----------------------BINARIZANDO A IMAGEM PARA QUE POSSAMOS UTILIZAR O MÉTODO QUE ENCONTRA OS CONTORNOS-------------------#

    sorce_image_binarized = cv2.adaptiveThreshold(sorce_image_morphology_operations, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5) 

    #----------------------ENCONTRANDO OS CONTORNOS NA IMAGEM BINARIZADA-------------------------------#

    contours,hierarquia = cv2.findContours(sorce_image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #----------------------CHAMANDO O MÉTODO FINDSHAPES QUE ENCONTRARÁ OS CONTORNOS DOS OBJETOS----------------------#

    #findShapes(contours, sorce_image, hierarquia)
    #shape_operations.findCont(contours, sorce_image, hierarquia)
    #figure.shape_operations.findContour(contours, sorce_image)
    figure.findContour(contours, sorce_image)

    #----------------------ESTRUTURANDO AS IMAGENS DE MODO QUE POSSAM SER USADAS EM UMA PILHA DE EXIBIÇÃO----------------------#

    #gray_image = np.stack((sorce_image_gray_image,)*3, axis=-1)

    sorce_image_binarized = np.stack((sorce_image_binarized,)*3, axis=-1)

    #images = [sorce_image_binarized ,sorce_image]
    presentation_imagens = [sorce_image_binarized ,sorce_image]

    img_stack = np.hstack(presentation_imagens) 

    cv2.imshow("IMAGEM-BINARIZADA / IMAGEM-CONTORNADA", img_stack)
    
    
    cv2.waitKey(frame_time) 