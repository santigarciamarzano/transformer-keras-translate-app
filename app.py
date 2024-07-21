import streamlit as st
import numpy as np
from keras_transformer import get_model
from scripts.data_preprocessing import load_dataset, tokenize_sentences
from scripts.tokenization import build_token_dict
from scripts.prepare_data import add_special_tokens
from scripts.model import build_transformer_model, translate


filename = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/english-spanish.pkl'
dataset = load_dataset(filename)
source_tokens, target_tokens = tokenize_sentences(dataset)

source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

encoder_input, decoder_input, output_decoded = add_special_tokens(source_tokens, target_tokens, source_token_dict, target_token_dict)

token_num = max(len(source_token_dict), len(target_token_dict))
model = build_transformer_model(token_num)

model_filepath = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/translator.weights.h5'
model.load_weights(model_filepath)

st.title('Traducción de Inglés a Español con Transformers')
st.write('Ingrese una frase en inglés y presione "Traducir" para obtener la traducción en español.')

sentence = st.text_input('Frase en inglés:')
if st.button('Traducir'):
    if sentence:
        translation = translate(model, sentence, source_token_dict, target_token_dict, target_token_dict_inv)
        st.write(f'Traducción: {translation}')
    else:
        st.write('Ingrese una frase para traducir (Sólo letra minúscula).')
