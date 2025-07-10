from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 训练过程
sentences = LineSentence('./data/malapi2019/all_analysis_data.txt')


model = Word2Vec(min_count=1, vector_size=128)

model.build_vocab(sentences)
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
word_vectors = model.wv
word_vectors.save("./checkpoints/malapiwv.wordvectors")

# 测试样例
wv = KeyedVectors.load("./checkpoints/malapiwv.wordvectors", mmap='r')
print("Vocabulary size:", len(wv.index_to_key))
vector = wv['ldrgetprocedureaddress']
print(vector)
print(vector.shape)


