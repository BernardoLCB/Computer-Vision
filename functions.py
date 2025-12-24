import cv2
import numpy as np

class image_operations:
    '''
    Classe dedicada ao processamento de imagens, abrangendo operações como aplicação de filtros de suavização e transformações morfológicas, com o objetivo de aprimorar ou analisar características visuais.
    '''
    
    def __init__(self):
        pass

    
    def SmoothingFilters(self, value, img):

    ### Método responsável por aplicar filtros de suavização na imagem de entrada ###
    
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


    def MorphologyOperations(self, img, slider_value):
        
    ### Função responsável por aplicar operações morfológicas na imagem de entrada.###
   
        value_matrix = (5,5)
        #kernel = np.ones(value_matrix , np.uint8)

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