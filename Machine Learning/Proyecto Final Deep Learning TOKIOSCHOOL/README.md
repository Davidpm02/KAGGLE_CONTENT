# PROYECTO FINAL DEEP LEARNING

## INTRODUCCIÓN

En este proyecto, se lleva a cabo el desarrollo completo de un modelo de visión artificial, que permite identificar imágenes de radiografías donde se puede apreciar o no la presencia de una fractura ósea. En este sentido, el proyecto contempla, desde el análisis y procesamiento de datos, hasta el entrenamiento, validacion y despliegue en producción del modelo final.

Previo al inicio del proyecto, se han definido una serie de hipótesis que se tratarán de refutar una vez contemos con los diferentes modelos entrenados, con el objetivo específico de tratar de responder a las hipótesis en cuestión.

El fichero REQUIREMENTS.txt incluye todas las libreras que se utilizan en el proyecto, incluyendo dentro de los ficheros Jupyter Notebook.

## CONJUNTO DE DATOS

El conjunto de datos que voy a utilizar lo he obtenido de Kaggle, a través de una publicación del usuario MADUSHANI RODRIGO, al que es posible acceder a través del siguiente enlace:
<https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data>.

Tal y como indica el propietario del dataset en la publicación del mismo, el conjunto de datos con el que vamos a trabajar consta de 3 subconjuntos de imágenes, donde contamos con:

* 9246 imágenes para el conjunto de entrenamiento.
* 828 imágenes para el conjunto de validación.
* 506 imágenes para el conjunto de prueba.

En total, el conjunto de datos cuenta con un total de 10,580 imágenes de radiografías, lo cual es una cantidad decente de partida para plantear el desarrollo de un modelo de visión artificial.

## HIPÓTESIS INICIALES

Como he mencionado, se han definido una serie de hipótesis o afirmaciones que deberán de ser contrastadas tras el desarrollo de los diferentes modelos. Este conjunto de hipótesis inicial nos ayuda a acercarnos al dataset, y a tratar de trabajar de diferentes maneras para ver como el rendimiento final de los modelos de visión artificial se ven afectados.

A continuación, se muestra el listado de hipótesis que se han definido para este proyecto:

* **Los patrones visuales en las radiografías de huesos fracturados son lo suficientemente distintos como para ser reconocidos por un modelo de Deep Learning.**

* **Puede que algunos patrones que distinguen huesos sanos de fracturados sean detectables por un modelo de Deep Learning, pero no sean tan fáciles de distinguir a simple vista.**

* **Un modelo de clasificación basado en redes neuronales convolucionales (CNN) puede aprender a diferenciar entre radiografías de huesos sanos y fracturados con alta precisión.**

* **La precisión en la clasificación de las imágenes, puede ser notablemente superior utilizando una red preentrenada.**

* **La calidad y cantidad de datos de entrenamiento (radiografías) influirá significativamente en el rendimiento del modelo.**

* **El modelo podrá generalizar bien a nuevas imágenes si se entrena con un conjunto diverso de radiografías que incluya diferentes tipos de fracturas, diferentes partes del cuerpo, y variabilidad en términos de calidad y ángulo de las imágenes.**

* **El preprocesamiento de las imágenes, como el ajuste de contraste, la normalización y el aumento de datos, mejorará la precisión del modelo.**

## EXTRAS

Junto con el modelo que se desplegará a producción a través del desarrollo de una API REST, se va a desarrollar un modelo YOLO de detección de entidades. Este modelo será accesible a través de una determirada ruta de la URI de la API con la que podremos interactuar con el modelo, y nos permitirá generar una imagen en donde se aprecia una determinada rotura, para una imagen que el modelo clasifique de manera positiva.

## Estructura del proyecto

A continuación, se incluye una breve descripción de la estructura del proyecto, y qué contiene cada directorio o archivo importante.

## Cómo Usar

Para poder hacer uso de este proyecto, simplemente debes seguir los siguientes pasos:

* Clona este repositorio.
* Navega al directorio del proyecto.
* Ejecuta Jupyter Notebook o cualquier otro IDE compatible para abrir los notebooks.
* Explora los notebooks que contienen el análisis de datos y la construcción de modelos.
* Para probar el modelo en producción, ejecuta el fichero "**run.py**" para inicializar la API REST. Interactua con el modelo navegando a través de las posibles rutas de la URI de la API.

## Contacto

Si tienes preguntas, sugerencias o te gustaría contribuir al proyecto, me encantaría escuchar tus ideas. Puedes contactar conmigo a traves de las siguientes maneras:

* ***Correo Electrónico***: <padishdev@duck.com>
* ***GitHub***: Si encuentras algun problema, o ves conveniente aplicar alguna modificacion, no dudes en abrir un issue. También puedes contribuir directamente mediante pull requests.

* ***LinkedIn***: <https://www.linkedin.com/in/david-padilla-mu%C3%B1oz-52126725a/>
