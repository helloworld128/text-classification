from gensim.models import word2vec, KeyedVectors
import numpy as np
import re
import torch
import torch.nn as nn

zh = re.compile(u'[^\u4e00-\u9fa5]')


def purge():
    with open('sina/sinanews.train', encoding='utf-8') as fp:
        out = open('sina/train.out', mode='w', encoding='utf-8')
        In = open('sina/train.in', mode='w', encoding='utf-8')
        for idx, l in enumerate(fp):
            tmp = []
            line = l.split()
            label = [line[1][6:]] + [line[i][3:] for i in range(2, 10)]
            out.write(' '.join(label) + '\n')
            for word in line[10:]:
                if not zh.search(word):
                    tmp.append(word)
            In.write(' '.join(tmp) + '\n')
            if idx % 100 == 0:
                print(idx)
        out.close()
        In.close()
    with open('sina/sinanews.test', encoding='utf-8') as fp:
        out = open('sina/test.out', mode='w', encoding='utf-8')
        In = open('sina/test.in', mode='w', encoding='utf-8')
        for idx, l in enumerate(fp):
            tmp = []
            line = l.split()
            label = [line[1][6:]] + [line[i][3:] for i in range(2, 10)]
            out.write(' '.join(label) + '\n')
            for word in line[10:]:
                if not zh.search(word):
                    tmp.append(word)
            In.write(' '.join(tmp) + '\n')
            if idx % 100 == 0:
                print(idx)
        out.close()
        In.close()


def generate():
    sentences = word2vec.LineSentence('sina/all.in')
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=80)
    model.wv.save_word2vec_format('data/model')


def check():
    model = KeyedVectors.load_word2vec_format('data/model')
    while 1:
        a, b = input().split()
        try:
            print(model.similarity(a, b))
            print(model.most_similar(a))
            print(model.most_similar(b))
        except:
            pass


if __name__ == '__main__':
    # generate()
    model = KeyedVectors.load_word2vec_format('data/model')

