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
- pip (para instalar dependencias)

## Instalación

1. Clona este repositorio:
    ```sh
    git clone https://github.com/tu_usuario/transformer-translation.git
    cd transformer-translation
    ```

2. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

### Preprocesamiento de Datos

1. Carga y tokeniza datos:
    ```python
    from data_preprocessing import load_dataset, tokenize_sentences

    filename = '/ruta/al/dataset/english-spanish.pkl'
    dataset = load_dataset(filename)
    source_tokens, target_tokens = tokenize_sentences(dataset)
    ```

2. Construye diccionarios de tokens:
    ```python
    from tokenization import build_token_dict

    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
    ```

3. Añade tokens especiales:
    ```python
    from prepare_data import add_special_tokens

    encoder_input, decoder_input, output_decoded = add_special_tokens(source_tokens, target_tokens, source_token_dict, target_token_dict)
    ```

### Modelo

1. Define y compila el modelo:
    ```python
    from model import build_transformer_model

    token_num = max(len(source_token_dict), len(target_token_dict))
    model = build_transformer_model(token_num)
    model.summary()
    ```

2. Entrena y guarda el modelo:
    ```python
    from model import train_and_save_model

    model_filepath = '/ruta/al/modelo/translator.h5'
    train_and_save_model(model, encoder_input, decoder_input, output_decoded, epochs=15, batch_size=32, filepath=model_filepath)
    ```

3. Carga el modelo y traduce una frase:
    ```python
    from model import translate

    model.load_weights(model_filepath)
    translate(model, 'the day is warm and sunny', source_token_dict, target_token_dict, target_token_dict_inv)
    ```

### Ejecución en el Notebook

Abre el notebook `train_custom.ipynb` y ejecuta las celdas paso a paso siguiendo el flujo descrito anteriormente.

### Ejecución en Servidor local

1. COrroborar que los archivos de pesos del modelo (translator.weights.h5) estén en la ruta correcta especificada en app.py

2. Ejecuta la aplicación de Streamlit:
   ```sh
   streamlit run app.py
   ```
3. La aplicación debería abrirse automáticamente en tu navegador predeterminado. Si no es así, abre tu navegador y navega a http://localhost:8501

