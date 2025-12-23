import cv2
import numpy as np
import sys
import os
from functions import findShapes, MorphologyOperations, SmoothingFilters

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
    cap = cv2.VideoCapture(caminho_videos+"/20.mp4")
    frame_time = 50
else:
    frame_time = 300

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
    
    elif choice == 1:
        #sorce_image = cv2.imread(caminho_imagem_base_com_lipobag+"/1.jpg")
        #sorce_image = cv2.imread(caminho_imagem_base_com_lipobag+"/2.jpg")
        sorce_image = cv2.imread(caminho_imagem_base_com_lipobag+"/3.jpg")
        
    

# else:

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/1.jpg") # 0/0/0/120/1/2000/2000/2000

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/2.jpg") # 2/0/0/128/1/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/3.jpg") # VALIDAR DEPOIS

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/4.jpg") # 1/0/0/0/1/2000/2000/2000
        
#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/5.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/6.jpg") # VALIDAR DEPOIS

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/7.jpg") # 1/0/0/69/2/2000/2000/+2000

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/8.jpg") # VALIDAR DEPOIS

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/9.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/10.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/11.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/13.jpg") # 2/0/0/0/1/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/14.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/15.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/16.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/17.png") # 0/0/0/2/2000/2000/2000

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/18.png") # 0/0/0/1/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/19.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/10.jpg") # NÃO SEI SE TEM VALIDAÇÃO
        
#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/21.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/22.jpg") # NÃO SEI SE TEM VALIDAÇÃO

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_dados + "/23.jpg") # 1/7/50/112/1/2000/2000/2000

#         # #--------------------------------------------------------------------------------------#

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/1.jpg") # 2/0/0/56/2/2000/2000/2000
        
#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/2.jpg") # 1/0/0/94/2/2000/2000/2000
        
#         #sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/3.jpg") # 0/0/0/62/2/1796/10000/10000/10000 ou 1/0/0/62/1/1796/2000/2000
        
#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/4.jpg") # 1/1/4/0/2/2000/2000/2000 ou 1/0/0/0/1/2000/10000/4000
        
#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/5.jpg") # 1/0/0/0/1/2000/2000/2000 ou 2/2/3/66/2/2000/2000/2000
        
#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/6.jpg") # 1/0/0/0/1/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/7.jpg") # 1/0/0/29/2/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/8.jpg") # 1/1/5/15/2/2000/2000/2000

#         # sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/9.jpg") # 1/3/14/23/2/2000/2000/2000

#         #sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/10.jpg") # 2/1/3/52/2/2000/2000/8728
        
#         #sorce_image = cv2.imread(caminho_imagem_exemplos_criados + "/11.jpg") # 0/0/0/0/2/2000/2000/2000
    

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

    ##---------------------CONVERTENDO A IMAGEM DE ENTRADA PARA O TOM DE CINZA#---------------------#
    sorce_image_gray_image = cv2.cvtColor(sorce_image, cv2.COLOR_BGR2GRAY)
    
    #---------------------APLICANDO FILTROS DE SUAVIZAÇÃO-------------------------------#
    
    sorce_image_smoothing_filter = SmoothingFilters(sliders1, sorce_image_gray_image)                            
    
    #-------------------------------APLICANDO FILTROS MORFOLÓGICOS-------------------------------#

    sorce_image_morphology_operations = MorphologyOperations(sorce_image_smoothing_filter, sliders2)

    #----------------------BINARIZANDO A IMAGEM PARA QUE POSSAMOS UTILIZAR O MÉTODO QUE ENCONTRA OS CONTORNOS-------------------#


    sorce_image_binarized = cv2.adaptiveThreshold(sorce_image_morphology_operations, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

    #_,sorce_image_binarized = cv2.threshold(sorce_image_morphology_operations, sliders5, 255, cv2.THRESH_BINARY) 

    #----------------------ENCONTRANDO OS CONTORNOS NA IMAGEM BINARIZADA-------------------------------#

    contours,hierarquia = cv2.findContours(sorce_image_binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #----------------------CHAMANDO O MÉTODO FINDSHAPES QUE ENCONTRARÁ OS CONTORNOS DOS OBJETOS----------------------#

    findShapes(contours, sorce_image, hierarquia)
    
    #----------------------ESTRUTURANDO AS IMAGENS DE MODO QUE POSSAM SER USADAS EM UMA PILHA DE EXIBIÇÃO----------------------#

    #gray_image = np.stack((sorce_image_gray_image,)*3, axis=-1)

    sorce_image_binarized = np.stack((sorce_image_binarized,)*3, axis=-1)

    #images = [sorce_image_binarized ,sorce_image]
    presentation_imagens = [sorce_image_binarized ,sorce_image]

    img_stack = np.hstack(presentation_imagens) 

    cv2.imshow("IMAGEM-BINARIZADA / IMAGEM-CONTORNADA", img_stack)
    
    
    cv2.waitKey(frame_time) 