def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

if __name__ == "__main__":
    from data_preprocessing import load_dataset, tokenize_sentences

    filename = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/english-spanish.pkl'
    dataset = load_dataset(filename)
    source_tokens, target_tokens = tokenize_sentences(dataset)

    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    print(source_token_dict)
    print(target_token_dict)
    print(target_token_dict_inv)