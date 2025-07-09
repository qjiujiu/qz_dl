
import os, sys
sys.path.append("./")
sys.path.append("../")


from utils import io


# 读取对抗向量的维度并做一个检查
if __name__ == '__main__':
    d = io.read_pickle(fname="data/malapi2019/emb-feature/LSTMTextClassifier/advexam-fgsm/train_adv_embeddings.pkl")
    X,y  = d
    print(len(X), X[0].shape)
    print(len(y))


    d = io.read_pickle(fname="data/malapi2019/emb-feature/LSTMTextClassifier/advexam-pgd/test_adv_embeddings.pkl")
    X,y  = d
    print(len(X), X[0].shape)
    print(len(y))

