# Mushroom Dataset (Binary Classification)

## Descripción del Proyecto
Este proyecto tiene como objetivo modelar y predecir la comestibilidad de las setas basándose en características físicas diversas. Utilizando el dataset "Secondary Mushroom" del UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset), este análisis pretende contribuir a la seguridad en la recolección de setas, diferenciando entre especies comestibles y venenosas.

### Dataset
El dataset "Secondary Mushroom" incluye las siguientes características detalladas:

#### Atributos
1. **cap-diameter (m)**: Diámetro del sombrero, número flotante en cm.
2. **cap-shape (n)**: Forma del sombrero (bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o).
3. **cap-surface (n)**: Superficie del sombrero (fibrous=i, grooves=g, scaly=y, smooth=s, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e).
4. **cap-color (n)**: Color del sombrero (brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k).
5. **does-bruise-bleed (n)**: Presencia de moretones o sangrado (bruises-or-bleeding=t, no=f).
6. **gill-attachment (n)**: Tipo de unión de las láminas (adnate=a, adnexed=x, decurrent=d, free=e, sinuate=s, pores=p, none=f, unknown=?).
7. **gill-spacing (n)**: Espaciado de las láminas (close=c, distant=d, none=f).
8. **gill-color (n)**: Color de las láminas, similar a los colores del sombrero + none=f.
9. **stem-height (m)**: Altura del tallo, número flotante en cm.
10. **stem-width (m)**: Ancho del tallo, número flotante en mm.
11. **stem-root (n)**: Tipo de raíz del tallo (bulbous=b, swollen=s, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r).
12. **stem-surface (n)**: Superficie del tallo, similar a la superficie del sombrero + none=f.
13. **stem-color (n)**: Color del tallo, similar a los colores del sombrero + none=f.
14. **veil-type (n)**: Tipo de velo (partial=p, universal=u).
15. **veil-color (n)**: Color del velo, similar a los colores del sombrero + none=f.
16. **has-ring (n)**: Presencia de anillo (ring=t, none=f).
17. **ring-type (n)**: Tipo de anillo (cobwebby=c, evanescent=e, flaring=r, grooved=g, large=l, pendant=p, sheathing=s, zone=z, scaly=y, movable=m, none=f, unknown=?).
18. **spore-print-color (n)**: Color de la impresión de esporas, similar a los colores del sombrero.
19. **habitat (n)**: Hábitat (grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d).
20. **season (n)**: Estación del año (spring=s, summer=u, autumn=a, winter=w).

#### Etiquetas de Clase
- **edible=e**: Comestible.
- **poisonous=p**: Venenoso.

## Requisitos
Para ejecutar los notebooks y scripts de este proyecto, necesitarás las siguientes herramientas y bibliotecas:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Tensorflow

Puedes instalar todas las dependencias necesarias con el siguiente comando:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```


## Estructura del Repositorio

Descripción breve de la estructura del repositorio y qué contiene cada directorio o archivo importante.

/


├── **data/**                 # Carpeta donde se almacenan los datos utilizados


├── **notebooks/**             # Jupyter Notebooks con análisis exploratorio y modelos


├── **src/**                   # Código fuente para scripts y módulos de Python


├── **tests/**                 # Pruebas unitarias para el código del proyecto


└── **README.md**


## Cómo Usar

Instrucciones detalladas sobre cómo ejecutar el código:

* Clona este repositorio.
* Navega al directorio del proyecto.
* Ejecuta Jupyter Notebook o cualquier otro IDE compatible para abrir los notebooks.
* Explora los notebooks que contienen el análisis de datos y la construcción de modelos.


## Contacto

Si tienes preguntas, sugerencias o te gustaría contribuir al proyecto, me encantaría escuchar tus ideas. Puedes contactar conmigo a traves de las siguientes maneras:

* ***Correo Electrónico***: padishdev@duck.com

* ***GitHub***: Si encuentras algun problema, o ves conveniente aplicar alguna modificacion, no dudes en abrir un issue. También puedes contribuir directamente mediante pull requests.

* ***LinkedIn***: https://www.linkedin.com/in/david-padilla-mu%C3%B1oz-52126725a/