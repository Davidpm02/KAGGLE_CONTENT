"""
Fichero Python que reune la logica que se utiliza en las distintas peticiones
que permite llevar a cabo la REST API.
"""

## IMPORTS -----


# MODELO
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# SERVIDOR API, MANEJO FICHEROS
from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename
import os

# PROCESAMIENTO IMAGENES 
import cv2 as cv
from PIL import Image, ImageDraw

# CARGA MODELO YOLO
import roboflow

# Funciones personalizadas para cargar y limpiar datos
from common.load_data import *
from common.clean_data import *

### Estas lineas ayudan a prevenir un error que se originaba al comenzar los entrenamientos
tf.data.experimental.enable_debug_mode()
tf.config.run_functions_eagerly(True)



## APP -----

app = Flask(__name__)



## FUNCTIONS -----

# Función para cargar el modelo de clasificación
def load_classification_model(file_name):
    
    """Funcion auxiliar encargada de cargar el modelo entrenado en memoria, y retornarlo al
       ambito global del programa.
       
       Args:
           file_name -- Ruta donde se aloja el modelo que queremos cargar.
        
       Returns:
           model -- Modelo cargado en memoria.
    """
    
    try:
        model = keras.models.load_model(file_name)
        return model
    except OSError as e:
        print(f"Error loading model: {e}")
        return None

# Cargo el modelo en memoria, y lo utilizo en el resto de funciones.
classification_model = load_classification_model(r"src\trained_models\residual_net_aug.h5")


def classify_image():
    """Función auxiliar encargada de procesar y utilizar el modelo para clasificar una imagen individual.
       Una vez se obtiene la predicción del modelo, esta se convierte a valor categórico nominal, y se retorna
       a la API REST, para que el usuario reciba el resultado.
       
       Args:
           
       Returns:
           results -- Diccionario con la predicción del modelo de clasificación.
           status_code -- Código de estado que indica a la API si la solicitud se ha procesado correctamente.
    """
    try:
        file = request.files.get('image')
        if file is None:
            return {"message": "No se ha proporcionado ninguna imagen."}, 400
        
        if classification_model is not None:
            try:
                # Me aseguro de que el directorio esté creado
                os.makedirs(r"src\uploads", exist_ok=True)
                filename = secure_filename(file.filename)
                filepath = os.path.join(r"src\uploads", filename)
                print("FILEPATH ==> ", filepath)
                file.save(filepath)

                # Procesamiento de la imagen
                processed_image = load_image_data(filepath, filename)

                # Predicción del modelo de clasificación
                prediction = classification_model.predict(processed_image)

                # Procesando la predicción para retornar una salida coherente al usuario
                result = "Fractured" if (prediction[0][0] >= 0.5) else "Not fractured"
                results = {
                    "classification": result,
                    "confidence": str(round(prediction[0][0], 5))
                }
                return results, 200
            except Exception as e:
                return {"message": f"Error al procesar la imagen: {str(e)}"}, 500
        else:
            return {"message": "No ha sido posible encontrar el modelo de clasificación."}, 500
    except Exception as e:
        return {"message": f"Error al procesar la solicitud: {str(e)}"}, 500

def classify_multiple_images():
    """Función auxiliar encargada de procesar y utilizar el modelo para clasificar múltiples imágenes contenidas
       en un array.
       Una vez se obtiene la predicción del modelo, esta se convierte a valor categórico nominal, y se retorna
       a la API REST, para que el usuario reciba el resultado.
       
       Args:
           
       Returns:
           results -- Diccionario con la predicción del modelo de clasificación para cada una de las imágenes recibidas.
           status_code -- Código de estado que indica a la API si la solicitud se ha procesado correctamente.
    """
    try:
        files = request.files.getlist('images')
        if not files:
            return {"message": "No se han proporcionado imágenes."}, 400

        if classification_model is not None:
            try:
                os.makedirs(r"src\uploads", exist_ok=True)
                results = {}
                for file in files:
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(r"src\uploads", filename)
                        print("FILEPATH ==> ", filepath)
                        file.save(filepath)

                        # Procesamiento de la imagen
                        processed_image = load_image_data(filepath, filename)

                        # Predicción del modelo de clasificación
                        prediction = classification_model.predict(processed_image)

                        # Procesando la predicción para retornar una salida coherente al usuario
                        result = "Fractured" if (prediction[0][0] >= 0.5) else "Not fractured"
                        results[filename] = {
                            "classification": result,
                            "confidence": str(round(prediction[0][0], 5))
                        }
                    except Exception as e:
                        results[file.filename] = {
                            "classification": "Error",
                            "confidence": "0",
                            "error": str(e)
                        }
                return results, 200
            except Exception as e:
                return {"message": f"Error al procesar las imágenes: {str(e)}"}, 500
        else:
            return {"message": "No ha sido posible encontrar el modelo de clasificación."}, 500
    except Exception as e:
        return {"message": f"Error al procesar la solicitud: {str(e)}"}, 500

def detect_fracture():
    """Función encargada de procesar una imagen individual con el modelo YOLO entrenado. 
       El modelo YOLO utilizado se encuentra desplegado en Roboflow, y se accede a este a través de
       la API de la plataforma.
       
       El modelo procesa la imagen y genera unos valores que corresponden a las coordenadas de la imagen,
       en donde este ha detectado la presencia de una fractura.
       
       Finalmente, se utilizan las coordenadas para marcar la fractura en la imagen, y se conserva una copia en local.
       
       Args:
       
       Returns:
           image -- Imagen procesada por el modelo, que incluye la detección de la fractura.
    """
    try:
        file = request.files.get('image')
        if file is None:
            return jsonify({"message": "No se ha proporcionado ninguna imagen."}), 400

        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(r"src\uploads", filename)
            os.makedirs(r"src\uploads", exist_ok=True)
            file.save(filepath)

            # Configurando la conexión a Roboflow
            rf = roboflow.Roboflow(api_key="hZBLpFFj2p0tR0gQNaLE")
            project = rf.workspace().project("fractured-on-xray-detection-zprrl")
            model = project.version("1").model

            # Umbrales de confianza y solapamiento
            model.confidence = 50  # Valores son porcentajes
            model.overlap = 50

            # Predicción del modelo 
            prediction = model.predict(filepath).json()['predictions']

            # Cargo la imagen original, para editarla ahora
            image = Image.open(filepath)
            draw = ImageDraw.Draw(image)

            # Dibujando las predicciones del modelo
            for detection in prediction:
                x, y, w, h = detection['x'], detection['y'], detection['width'], detection['height']
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2
                draw.rectangle([left, top, right, bottom], outline="red", width=2)
                draw.text((left, top), detection['class'], fill="red")

            # Guardo la imagen con las detecciones
            detection_image_path = r"src\uploads\detection.jpeg"
            image.save(detection_image_path)

            return send_file("../src/uploads/detection.jpeg", mimetype='image/jpeg')
        
        except Exception as e:
            return jsonify({"message": f"Error al procesar la imagen: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"message": f"Error al procesar la solicitud: {str(e)}"}), 500


def fine_tune_endpoint():
    
    """
        Punto final para ajustar el clasificador y entrenar el modelo al que accede el usuario a traves
        de la API para iniciar el proceso de fine-tuning del modelo.

        Returns:
            response: Tupla que contiene la respuesta JSON y el código de estado.
                - metrics (dict): El diccionario de respuesta que contiene las métricas de clasificacion
                                  obtenidas por el modelo.
                - status (int): El código de estado que indica el éxito o fracaso del proceso.
        """
    try:
        fine_tune_classifier_uploads()
        metrics = train_model()
        response = {
            "status": "Modelo reentrenado con éxito",
            "metrics": metrics
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

def fine_tune_classifier_uploads():
    
    """Funcion auxiliar encargada de clasificar las imagenes del directorio 'src/uploads' para 
       su posterior uso en la creacion de los conjuntos de entrenamiento, validacion y prueba.
       
       Las imagenes, une vez clasificadas, son almacenadas en el directorio 'src\data_finetuning_uploads',
       dentro de las subcarpetas 'fractured' o 'non_fractured', respectivamente.
       
       Args:
           
       Returns:
        
    """
    fine_tune_path = r"src\data_finetuning_uploads"
    os.makedirs(fine_tune_path, exist_ok=True)
    os.makedirs(os.path.join(fine_tune_path, "fractured"), exist_ok=True)
    os.makedirs(os.path.join(fine_tune_path, "non_fractured"), exist_ok=True)

    for image_name in os.listdir(r"src\uploads"):
        image_path = os.path.join(r"src\uploads", image_name)
        print(image_path)
        image = cv.imread(image_path)
        if image is None:
            print(f"Error al cargar la imagen: {image_name}")
            continue

        processed_image = load_image_data(image_path, image_name)
        prediction = classification_model.predict(processed_image)

        result = "fractured" if prediction >= 0.5 else "non_fractured"
        destination = os.path.join(fine_tune_path, result, image_name)
        try:
            os.rename(image_path, destination)
        except Exception as e:
            print(f"La imagen {image_name} ya se encontraba en el directorio indicado. Error: {e}")
            continue
        print(f"La instancia '{image_name}' se ha incluido en el conjunto de instancias {result}.")

    create_train_test_splits(fine_tune_path)

def create_train_test_splits(fine_tune_path, train_split=0.8):
    
    """Funcion auxiliar encargada de procesar la separacion de instancias, en lo que formaran los nuevos conjuntos de datos,
       que puedan ser cargados con la clase ImageDataGenerator.
       
       Args:
           fine_tune_path -- Ruta del directorio que contiene las imagenes que se utilizan durante todo el proceso de fine-tuning.
           train_split -- Float que representa el porcentaje de imagenes que van a corresponder al conjunto de entrenamiento.
       Returns:
        
    """
    
    train_dir = os.path.join(fine_tune_path, "train")
    test_dir = os.path.join(fine_tune_path, "test")
    os.makedirs(os.path.join(train_dir, "fractured"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "non_fractured"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "fractured"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "non_fractured"), exist_ok=True)

    def move_files(src_dir, train_dest_dir, test_dest_dir):
        files = os.listdir(src_dir)
        n_train = int(len(files) * train_split)
        for index, file_name in enumerate(files):
            src_file = os.path.join(src_dir, file_name)
            if index < n_train:
                dst_file = os.path.join(train_dest_dir, file_name)
            else:
                dst_file = os.path.join(test_dest_dir, file_name)
            os.rename(src_file, dst_file)

    move_files(os.path.join(fine_tune_path, "fractured"), os.path.join(train_dir, "fractured"), os.path.join(test_dir, "fractured"))
    move_files(os.path.join(fine_tune_path, "non_fractured"), os.path.join(train_dir, "non_fractured"), os.path.join(test_dir, "non_fractured"))

def train_model():
    
    """Funcion auxiliar encargade de procesar todo el entrenamiento del modelo. La funcion carga los conjuntos de datos
       mediante una llamada a la funcion 'generate_subsets', y define algunos hiperpararametros con los ajustar el entrenamiento
       del modelo.
       El modelo entrenado corresponde a la red residual que, al terminar el entrenamiento, es almacenada en el mismo lugar,
       sobreescribiendose.
       
       Args:
       
       Returns:
           metrics -- Diccionario con las metricas de clasificacion obtenidas por el modelo, al ser evaluado frente al conjunto
                      de prueba inicial.
    """
    
    try:
        # Defino los conjuntos de datos
        train_data, val_data, test_data = generate_subsets()
        
        batch_size = 20
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
            )
        model_checkpoint = ModelCheckpoint(
            r"src\trained_models\residual_net_aug.h5",
            save_best_only=True,
            monitor='val_loss'
            )

        # Compilo el modelo (activo run_eagerly=True debido a un error que persistia al iniciar los entrenamientos)
        classification_model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='binary_crossentropy',
            metrics=['accuracy'],
            run_eagerly=True
            )
        
        
        steps_per_epoch = max(train_data.samples // batch_size, 1)
        validation_steps = max(val_data.samples // batch_size, 1)

        # Entrenamiento del modelo
        classification_model.fit(
            train_data,
            validation_data=val_data,
            epochs=3,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stopping, model_checkpoint]
        )

        # Genero predicciones para el conjunto de prueba
        y_pred = classification_model.predict(test_data)
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

        # Construyo el diccionario con las metricas de clasificacion
        metrics = {
            "f1_score": round(f1_score(test_data.classes, y_pred), 3),
            "recall_score": round(recall_score(test_data.classes, y_pred), 3),
            "precision_score": round(precision_score(test_data.classes, y_pred), 3),
            "accuracy_score": round(accuracy_score(test_data.classes, y_pred), 3)
        }

        classification_model.save(r"src\trained_models\residual_net_aug.h5")
        return metrics
    except Exception as e:
        return {"Error en el proceso de entrenamiento": str(e)}

def generate_subsets():
    
    """Funcion auxiliar encargada de instanciar dos generadores de la clase ImageDataGenerator, 
       y definir los conjuntos de datos con los que entrenar el modelo.
       
       Args:
       
       Returns:
           train_data -- Conjunto de datos de entrenamiento.
           val_data -- Conjunto de datos de validacion.
           test_data -- Conjunto de datos de prueba.
    """
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    datagen_train = ImageDataGenerator(rescale=1. / 255, validation_split=0.10)
    fine_tune_path = r"src\data_finetuning_uploads"
    batch_size = 20

    train_path = os.path.join(fine_tune_path, "train")
    train_data = datagen_train.flow_from_directory(
        directory=train_path,
        target_size=(224, 224),
        color_mode="rgb",
        classes=["non_fractured", "fractured"],
        shuffle=True,
        batch_size=batch_size,
        class_mode="binary",
        subset='training'
    )

    val_data = datagen_train.flow_from_directory(
        directory=train_path,
        target_size=(224, 224),
        color_mode="rgb",
        classes=["non_fractured", "fractured"],
        shuffle=True,
        batch_size=batch_size,
        class_mode="binary",
        subset='validation'
    )

    test_path = os.path.join(r"data", "test")
    test_data = datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        color_mode="rgb",
        classes=["non_fractured", "fractured"],
        shuffle=False,
        batch_size=batch_size,
        class_mode="binary"
    )

    return train_data, val_data, test_data