import cv2
import numpy as np


class shape_operations:

    def __init__(self):
        pass


    def findContour(self, contours, img):
    ### Metodo responsável por determinar a forma geométrica com base no contorno já encontrado ###

        min_area = 1000

        for contour in contours:
        
            approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)

            name_contour = None
            color = None

            if (len(approx) == 3 and cv2.contourArea(contour) >= min_area):
                name_contour = "Triangle"
                color = (0, 255, 0) # verde
            
            elif (len(approx) == 5 and cv2.contourArea(contour) >= min_area):
                name_contour = "Pentagon"
                color = (0, 0, 255) # vermelho
            
            elif (len(approx) == 10 and cv2.contourArea(contour) >= min_area):
                name_contour = "Star"
                color = (255, 0, 0) # azul
            
            if name_contour in ["Triangle", "Pentagon", "Star"]:
                self.draw_contour(name_contour, color, approx, img, contour)


    def draw_contour(self, name_contour, color, approx, img, contour):
    ### Método responsável por exibir os contornos na imagem ###
            
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