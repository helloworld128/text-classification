import numpy as np
import pickle
import sys


datadir = 'data/'
inputFile = datadir + 'sgns.sogounews.bigram-char'
dataFile = datadir + 'data.npy'
out_words = datadir + 'words.pkl'
out_idx = datadir + 'idx.pkl'
vocab_size = 365200
embedding_dim = 300


def parse_data():
    words = []
    idx = 0
    word2idx = {}
    vectors = np.zeros((vocab_size, embedding_dim), dtype=np.float)

    with open(inputFile, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.split()
            if len(line) == 302:  # extra spaces
                vectors[idx, :] = line[2:]
                word = line[0] + line[1]
                word2idx[word] = idx
                words.append(word)
                idx += 1
            elif len(line) == 301:  # normal
                vectors[idx, :] = line[1:]
                word = line[0]
                word2idx[word] = idx
                words.append(word)
                idx += 1
    pickle.dump(words, open(out_words, 'wb'))
    pickle.dump(word2idx, open(out_idx, 'wb'))
    np.save(dataFile, vectors)
    print('parse finished, %d words altogether.' % idx)


def load():
    vectors = np.load(dataFile)
    words = pickle.load(open(out_words, 'rb'))
    word2idx = pickle.load(open(out_idx, 'rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return vectors, words, word2idx, glove


def peek():
    vectors, words, word2idx, glove = load()
    while 1:
        x, y = input().split()
        try:
            vx, vy = glove[x], glove[y]
            print(vx.dot(vy) / (np.sqrt(vx.dot(vx) * vy.dot(vy))))
        except KeyError:
            print('word does not exist.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('begin parsing.')
        parse_data()
    elif sys.argv[1] == 'test':
        peek()

