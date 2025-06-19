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

from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.get_config import load_config

def create_model(config):
    return LSTMTextClassifier(
        vocab_size=278,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    )

config_path = "config/clean_adv_config.yaml"  
config = load_config(config_path)
model = create_model(config)
for name in model.state_dict().keys():
    print(name)
