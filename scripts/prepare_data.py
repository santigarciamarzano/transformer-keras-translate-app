def add_special_tokens(source_tokens, target_tokens, source_token_dict, target_token_dict):
    encoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
    decoder_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
    output_tokens = [tokens + ['<END>'] for tokens in target_tokens]

    source_max_len = max(map(len, encoder_tokens))
    target_max_len = max(map(len, decoder_tokens))

    encoder_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encoder_tokens]
    decoder_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decoder_tokens]
    output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

    encoder_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encoder_tokens]
    decoder_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decoder_tokens]
    output_decoded = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

    return encoder_input, decoder_input, output_decoded

if __name__ == "__main__":
    from data_preprocessing import load_dataset, tokenize_sentences
    from tokenization import build_token_dict

    filename = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/english-spanish.pkl'
    dataset = load_dataset(filename)
    source_tokens, target_tokens = tokenize_sentences(dataset)

    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    encoder_input, decoder_input, output_decoded = add_special_tokens(source_tokens, target_tokens, source_token_dict, target_token_dict)

    print(encoder_input[120000])
