"""
Fichero Python en el que se definen diferentes funciones que poder utilizar
para llevar a cabo el proceso de carga de datos.

Se contempla que las imagenes seran cargadas de una en una en la API REST para 
entregarlas posteriormente al modelo, por lo que, en esencia, se define
una funcion para llevar a cabo esta carga de datos.
"""


# IMPORTS -----

import numpy as np
import cv2 as cv
import os
from .clean_data import reshape_data, flatten_data


# FUNCTIONS -----

def load_image_data(data_path):
    
    """Funcion parametrizada encargada de cargar en memoria una instancia para poder enviarla a los 
       modelos desarrollados, y dejarla lista para ser procesada por estos.
       
       Args:
           data_path -- Ruta donde se encuentra la instancia a cargar en memoria.
       Returns:
       
    """
    
    # Cargo la imagen memoria
    image = cv.imread(data_path)
    
    ## PROCESAMIENTO ---
    # Redimensiono la instancia a 224x224
    imagen_resized = reshape_data(image)
    
    # Asigno una escala de grises a la imagen
    image_gray = flatten_data(imagen_resized)
    
    # Almaceno la imagen en el directorio correspondiente
    instance_api_path = "../src/image_api/"
    
    if not os.path.exists(instance_api_path):
        os.mkdir(instance_api_path)
        
    # Conservo la copia dentro del directorio indicado
    cv.imwrite(os.path.join(instance_api_path, "img_to_api.jpg"), image_gray)
    
    
    
    