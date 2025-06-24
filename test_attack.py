# import torch
# from models.cv.lenet import LeNet

# def test_attack():
#     model = LeNet(num_classes=10)
#     x = torch.randn(32, 1, 28, 28)
    
#     y = model(x)

#     print(y.size())

#     z = model.fgsm_attack(x, y, epsilon=0.01)
#     print(z.size())
    

# test_attack()

# 打印出模型包含的所有权重
# from models.nlp.lstm_text_classifier import LSTMTextClassifier
# from utils.get_config import load_config

# def create_model(config):
#     return LSTMTextClassifier(
#         vocab_size=278,
#         embedding_dim=config['embedding_dim'],
#         hidden_dim=config['hidden_dim'],
#         output_dim=config['output_dim'],
#         max_len=config['max_len']
#     )

# config_path = "config/clean_adv_config.yaml"  
# config = load_config(config_path)
# model = create_model(config)
# for name in model.state_dict().keys():
#     print(name)

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)
print(embeddings.shape)