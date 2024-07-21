# Traducción de Inglés a Español con Transformers

Este proyecto utiliza un modelo Transformer para traducir frases del inglés al español. El código incluye el preprocesamiento de datos, la construcción del modelo, el entrenamiento y la traducción de frases.

## Estructura del Proyecto

- `data_preprocessing.py`: Carga y tokeniza los datos del dataset.
- `tokenization.py`: Construye diccionarios de tokens.
- `prepare_data.py`: Añade tokens especiales a las frases.
- `model.py`: Define y entrena el modelo Transformer, además de traducir frases.
- `notebook.ipynb`: Notebook para ejecutar todo el flujo del proyecto.
- `requirements.txt`: Archivo con las dependencias necesarias.

## Requisitos

- Python 3.10
- pip 

## Instalación

1. Clona este repositorio:
    ```sh
    git clone https://github.com/santigarciamarzano/transformer-keras-translate-machine-app.git
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```
## Archivos grandes

Debido a las limitaciones de tamaño de archivo de GitHub, los archivos entrenados y los datos están disponibles para descarga en los siguientes enlaces:

- [Modelo entrenado (.h5)](https://drive.google.com/uc?export=download&id=1m_6kVg1OwurfUzEZaIkiPlfrm1A6QBYb)
- [Datos (.pkl)](https://drive.google.com/uc?export=download&id=1tqevUH5Hy4wFGKgqX2Du3Djw7xEF3wZ8)

## Uso

### Preprocesamiento de Datos

1. Carga y tokeniza datos
    
2. Construye diccionarios de tokens
  
3. Añade tokens especiales
   
### Modelo

1. Define y compila el modelo:
   
2. Entrena y guarda el modelo:
   
3. Carga el modelo y traduce una frase:
 
### Ejecución en el Notebook

Abre el notebook `train_custom.ipynb` y ejecuta las celdas paso a paso siguiendo el flujo descrito anteriormente.

### Ejecución en Servidor local

1. COrroborar que los archivos de pesos del modelo (translator.weights.h5) estén en la ruta correcta especificada en app.py

2. Ejecuta la aplicación de Streamlit:
   ```sh
   streamlit run app.py
   ```
3. La aplicación debería abrirse automáticamente en tu navegador predeterminado. Si no es así, abre tu navegador y navega a http://localhost:8501

