from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
from tqdm import tqdm

sentences = LineSentence('./data/malapi2019/all_analysis_data.txt')

model = FastText(vector_size=128, window=3, min_count=1)
model.build_vocab(corpus_iterable=sentences)
# 训练，添加 tqdm 进度条（按 epoch）
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    # 包一层 tqdm，每轮重新读取语料，显示处理进度
    model.train(
        corpus_iterable=tqdm(sentences, desc=f"Training Epoch {epoch+1}"),
        total_examples=model.corpus_count,
        epochs=1
    )

# model.train(corpus_iterable=sentences, total_examples=model.corpus_count, epochs=10)
word_vectors = model.wv
word_vectors.save("./checkpoints/malapift.wordvectors")

# 测试样例
wv = KeyedVectors.load("./checkpoints/malapift.wordvectors", mmap='r')
print("Vocabulary size:", len(wv.index_to_key))
vector = wv['ldrgetprocedureaddress']
print(vector)
print(vector.shape)