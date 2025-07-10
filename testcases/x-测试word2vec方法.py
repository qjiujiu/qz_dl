from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


if __name__ == "__main__":
    sentences = LineSentence('./data/malapi2019/all_analysis_data.txt')

    model = Word2Vec(min_count=1, vector_size=128)
    model.build_vocab(sentences) 
    # 建立词表

    wv: KeyedVectors = model.wv
    # wv.key_to_index
    # wv.index_to_key

    key = wv.index_to_key[1]
    vec = wv[key]

    print(vec)
    print(vec.shape)


