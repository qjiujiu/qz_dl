import torch
import torch.nn as nn

import os, sys
sys.path.append("./")
sys.path.append("../")

from models.nlp.lstm_text_adv import LSTMTextAdvAttnClassifier
from models.nlp.lstm_text_classifier import LSTMTextAttnClassifier
from models.nlp.tcn import TCNTextClassifier
from models.nlp.atten import MLPAttention, SelfAttention

from config.logger import logger

logger.is_debug(True)


def test_attn():
    # embedded.shape: torch.Size([8, 200, 128])
    r = torch.randn([8, 200, 128])
    mlp_attn = MLPAttention(embed_dim=128)
    self_atten = SelfAttention(embed_dim=128)
    
    logger.debug(f"受试模型: {mlp_attn.__class__.__name__}")
    logger.debug(r.shape)
    logger.debug(mlp_attn(r).shape)
    logger.debug(self_atten(r).shape)



def test_tcn():
    vocab_size = 178
    batch_size, seq_len = 8, 200
    r = torch.randint(0, vocab_size, (batch_size, seq_len))
    tcn = TCNTextClassifier(vocab_size=178, embedding_dim=128, tcn_channels=[128, 128], output_dim=8)

    logger.debug(f"受试模型: {tcn.__class__.__name__}")
    logger.debug(r.shape)
    logger.debug(tcn(r).shape) 


def test_attn_lstm():
    vocab_size = 178
    batch_size, seq_len = 8, 200
    r = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = LSTMTextAttnClassifier(
        vocab_size=178, embedding_dim=128, hidden_dim=256,
        bidirectional=True, output_dim=8, layers=1
    )

    z = model.embed(r)
    y = model.forward(z)
    logger.debug(f"受试模型: {model.__class__.__name__}")
    logger.debug(f"r = {r.shape}")
    logger.debug(f"z = {z.shape}") 
    logger.debug(f"y = {y.shape}") 

       


if __name__ == '__main__':
    test_attn()
    # test_tcn()
    # test_attn_lstm()