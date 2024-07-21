from keras_transformer import get_model, decode
import numpy as np

def build_transformer_model(token_num, embed_dim=32, encoder_num=2, decoder_num=2, head_num=4, hidden_dim=128, dropout_rate=0.05):
    model = get_model(
        token_num=token_num,
        embed_dim=embed_dim,
        encoder_num=encoder_num,
        decoder_num=decoder_num,
        head_num=head_num,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        use_same_embed=False,
    )
    model.compile('adam', 'sparse_categorical_crossentropy')
    return model

def train_and_save_model(model, encoder_input, decoder_input, output_decoded, epochs=15, batch_size=32, filepath='/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/translator.weights.h5'):
    x = [np.array(encoder_input), np.array(decoder_input)]
    y = np.array(output_decoded)
    model.fit(x, y, epochs=epochs, batch_size=batch_size)
    model.save_weights(filepath)
    return model

'''def translate(model, sentence, source_token_dict, target_token_dict, target_token_dict_inv):
    sentence_tokens = [tokens + ['<END>', '<PAD>'] for tokens in [sentence.split(' ')]]
    tr_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in sentence_tokens][0]
    decoded = decode(
        model, 
        tr_input, 
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>']
    )

    print('Frase original: {}'.format(sentence))
    print('Traducción: {}'.format(' '.join(map(lambda x: target_token_dict_inv[x], decoded[1:-1])))) '''

def translate(model, sentence, source_token_dict, target_token_dict, target_token_dict_inv):
    sentence_tokens = [sentence.split(' ')]
    sentence_tokens = [tokens + ['<END>'] + ['<PAD>']*(len(source_token_dict) - len(tokens) - 1) for tokens in sentence_tokens]
    tr_input = [list(map(lambda x: source_token_dict.get(x, source_token_dict['<PAD>']), tokens)) for tokens in sentence_tokens][0]

    decoded = decode(
        model, 
        tr_input, 
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>']
    )

    translation = ' '.join(map(lambda x: target_token_dict_inv.get(x, ''), decoded[1:-1]))
    return translation


if __name__ == "__main__":
    from tokenization import build_token_dict
    from data_preprocessing import load_dataset, tokenize_sentences
    from prepare_data import add_special_tokens

    filename = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/english-spanish.pkl'
    dataset = load_dataset(filename)
    source_tokens, target_tokens = tokenize_sentences(dataset)

    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    encoder_input, decoder_input, output_decoded = add_special_tokens(source_tokens, target_tokens, source_token_dict, target_token_dict)

    token_num = max(len(source_token_dict), len(target_token_dict))
    model = build_transformer_model(token_num)

    train_and_save_model(model, encoder_input, decoder_input, output_decoded, filepath='/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/translator.h5')

    translate(model, 'the day is warm and sunny', source_token_dict, target_token_dict, target_token_dict_inv)
