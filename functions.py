import cv2
import numpy as np


def SmoothingFilters(value,img):
    
#========================================================================================#
# Função responsável por aplicar filtros de suavização na imagem de entrada#
#========================================================================================#

    #FILTRO DE GAUSS
    if value == 1:
        mask = (5,5)
        smoothing = 0
        img = cv2.GaussianBlur(img, mask, smoothing)
    
    #FILTRO DE MÉDIA
    elif value == 2:
        mask = (5,5)
        img = cv2.blur(img, mask)
    
    #FILTRO DE MEDIANA
    elif value == 3:
        smoothing = 5
        img = cv2.medianBlur(img,smoothing)
    
    #FILTRO DE CANNY
    elif value == 4:
        thresold1 = 30    
        thresold2 = 120 
        maskSobel = 3 #valor padrão             
        img = cv2.Canny(img, thresold1, thresold2, maskSobel)
    
    return img


def MorphologyOperations(img, slider_value):

#========================================================================================#
# Função responsável por aplicar operações morfológicas na imagem de entrada.#
#========================================================================================#

    value_matrix = (5,5)
    kernel = np.ones(value_matrix , np.uint8)

    if slider_value != 0:

        #erosão: remove ruídos e pequenas estruturas, reduzindo o tamanho dos objetos brancos na imagem.
        if slider_value == 1:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.erode(img, element_estr, iterations = 2)

        #dilatação: expande as regiões brancas da imagem, preenchendo pequenos buracos.
        elif slider_value == 2:
            element_estr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, value_matrix)
            img = cv2.dilate(img, element_estr, iterations=2)

        #abertura: realiza erosão seguida de dilatação, útil para remover pequenos ruídos.
        elif slider_value == 3:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element_estr)
        
        #fechamento: realiza dilatação seguida de erosão, útil para fechar buracos em objetos.
        elif slider_value == 4:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element_estr)

        #gradiente: obtém o contorno dos objetos ao calcular a diferença entre a dilatação e a erosão.
        elif slider_value == 5:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, element_estr)
        
        #top-hat: realça detalhes menores e variações de intensidade na imagem.
        elif slider_value == 6:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, element_estr)
            
        #black-Hat: destaca regiões escuras em fundos claros.
        elif slider_value == 7:
            element_estr = cv2.getStructuringElement(cv2.MORPH_RECT, value_matrix)
            img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, element_estr)


    return img


def PoseEstimation(contorno = "opcional"):

    # # definindo os pontos 3d do objeto de interesse
    # dimensoes da lipogab: largura = 19,5cm = 0,195m ; comprimento = 8cm = 0,08m e altura = 7cm = 0,07m

    pontos_3d_objeto = np.array([

        [0,0,0]     # coordenada 1 INFERIOR-ESQUERDO (olhando de cima)
        [0,0,0]     # coordenada 2 INFERIOR-DIREITO (olhando de cima)
        [0,0,0]     # coordenada 3 SUPERIOR-DIREITO (olhando de cima)
        [0,0,0]     # coordenada 4 SUPERIOR-ESQUERDO (olhando de cima)

    ], dtype=np.float32)

    #x, y, w, h = cv2.boundingRect(contorno)

    # pontos_2d_detectados = np.array([

    #     [x, y]     # ponto onde a coordenada 1 aparece na imagem
    #     [x+w, y]     # ponto onde a coordenada 2 aparece na imagem
    #     [x+w, y+h]     # ponto onde a coordenada 3 aparece na imagem
    #     [x, y+h]     # ponto onde a coordenada 4 aparece na imagem

    # ], dtaype = np.float32)
    
    pontos_2d_detectados = np.array([

        [x1,y1]     # ponto onde a coordenada 1 aparece na imagem
        [x2,y2]     # ponto onde a coordenada 2 aparece na imagem
        [x3,y3]     # ponto onde a coordenada 3 aparece na imagem
        [x4,y4]     # ponto onde a coordenada 4 aparece na imagem

    ], dtaype = np.float32)

    success, rvec, tvec = cv2.solvePnP(
        pontos_3d_objeto,
        pontos_2d_detectados,
        camera_matriz,
        dis_coeffs
        )
    
    if success:
        print(f"Posição: X={tvec[0][0]:.2f}m, Y={tvec[1][0]:.2f}m, Z={tvec[2][0]:.2f}m")
        print(f"Distância: {tvec[2][0]:.2f} metros")

    pass
