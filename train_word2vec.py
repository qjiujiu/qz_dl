from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 训练过程
sentences = LineSentence('./data/malapi2019/all_analysis_data.txt')
# sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

# print(sentences)
# for sentence in sentences:
#     print(sentence)
#     break

model = Word2Vec(min_count=1)
model.build_vocab(sentences)  # prepare the model vocabulary
model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
word_vectors = model.wv
word_vectors.save("./word2vec.wordvectors")

# # 测试样例
wv = KeyedVectors.load("./word2vec.wordvectors", mmap='r')
print("Vocabulary size:", len(wv.index_to_key))
vector = wv['ldrgetprocedureaddress']
print(vector)


