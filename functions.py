from sre_constants import SUCCESS
import cv2
import numpy as np


def findContour(contour):

    min_area = 1000
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

    name_contour = None
    color = None

    if (len(approx) == 3 and cv2.contourArea(contour) >= min_area):
        name_contour = "Triangle"
        color = (0, 255, 0) #verde
    
    elif (len(approx) == 5 and cv2.contourArea(contour) >= min_area):
        name_contour = "Pentagon"
        color = (0, 0, 255) #vermelho
    
    elif (len(approx) == 10 and cv2.contourArea(contour) >= min_area):
        name_contour = "Star"
        color = (255, 0, 0) #azul


    return (name_contour, color, approx)


#========================================================================================#
'''Função responsável por determinar a forma geométrica com base no contorno já encontrado.'''
#========================================================================================#

def findShapes(contours, img, hierarquia):

    #============================================================================#

    def draw_contour(name_contour, color, approx, img, contour):
        
        x, y, _, _ = cv2.boundingRect(approx)

        #verifica a existência do objeto com base na quantidade de pixels que possui
        centro_x = None
        centro_y = None 

        M = cv2.moments(contour)
        if M["m00"] != 0:   # Evita divisão por zero
            centro_x = int(M["m10"] / M["m00"])
            centro_y = int(M["m01"] / M["m00"])

        cv2.drawContours(img, [approx], -1, color, 4)
        cv2.putText(img, name_contour, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, color)

        if (centro_x != None) and (centro_y != None):
            cv2.circle(img,(centro_x, centro_y), 5, color ,-1)
    
    #============================================================================#
    

    for i, contour in enumerate(contours):
        name_contour, color, approx = findContour(contour)
        choices = ["Triangle", "Pentagon", "Star"]
        if name_contour in choices:
            draw_contour(name_contour, color, approx, img, contour)



#========================================================================================#
'''Função responsável por aplicar filtros de suavização na imagem de entrada'''
#========================================================================================#

def SmoothingFilters(value,img):

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


#========================================================================================#
'''Função responsável por aplicar operações morfológicas na imagem de entrada.'''
#========================================================================================#

def MorphologyOperations(img, slider_value):

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
