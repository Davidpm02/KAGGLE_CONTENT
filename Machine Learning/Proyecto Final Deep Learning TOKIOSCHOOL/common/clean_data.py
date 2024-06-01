"""
Fichero Python que contiene el codigo necesario para llevar a cabo la limipieza
y preparacion de los datos, con el objetivo de poder utilizarlos en los modelos
desarrollados.
"""

# IMPORTS -----

# IMPORTS ------

import numpy as np
import cv2 as cv
import os


# FUNCTIONS -----

def reshape_data(image):
    
    """Funcion encargada de procesar una instancia de datos y cambiar su resolucion a
       a una que sea compatible con el modelo desplegado en la API REST.
       
       Args:
           image -- Instancia del conjunto de datos a procesar.
       Return:
           image -- Instancia procesada.
    """
    
    # Instancio una nueva variable, que contenga la imagen procesada a 224x224
    image = cv.resize(image, (224, 224))
    return image


def flatten_data(image):
    
    """Funcion encargada de procesar una instancia de datos y asignarle una escala
       de grises. Esta es el esquema que esperan los modelos desplegados en la API REST.
       
       Args:
           image -- Instancia del conjunto de datos a procesar.
       Return:
           image -- Instancia procesada.
    """
    
    # Instancio una nueva variable, a la que le aplico el cambio en los canales
    # de color
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return img_gray