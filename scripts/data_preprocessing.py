import numpy as np
from pickle import load

def load_dataset(filepath):
    return load(open(filepath, 'rb'))

def tokenize_sentences(dataset):
    source_tokens = [sentence.split(' ') for sentence in dataset[:, 0]]
    target_tokens = [sentence.split(' ') for sentence in dataset[:, 1]]
    return source_tokens, target_tokens

if __name__ == "__main__":
    
    filename = '/media/minigo/Disco/modelado3d/santiago/Capacitaciones/Keras Transformers/english-spanish.pkl'
    dataset = load_dataset(filename)
    print(dataset[120000:120010, 0])
    print(dataset[120000:120010, 1])

    source_tokens, target_tokens = tokenize_sentences(dataset)
    print(source_tokens[120000])
    print(target_tokens[120000])
