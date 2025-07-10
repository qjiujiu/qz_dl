from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
from tqdm import tqdm


if __name__ == '__main__':
    sentences = LineSentence('./data/malapi2019/all_analysis_data.txt')
    model = FastText(vector_size=128, window=3, min_count=1)
    model.build_vocab(corpus_iterable=sentences)


    wv: KeyedVectors = model.wv
    # wv.key_to_index
    # wv.index_to_key

    key = wv.index_to_key[1]
    vec = wv[key]

    print(vec)
    print(vec.shape)


