import torch
import torch.nn as nn
import numpy as np

from models.nlp.conv_text import (
    Conv1dTextClassifier,
    Conv2dTextClassifier
)
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from models.nlp.lstm_text_adv import LSTMTextAdvClassifier
from models.nlp.tcn import TCNTextClassifier


from config.params_parser.params_template import (
    CommonCfgParams, 
    NlpCfgParams,
    CvCfgParams
)
from utils import io
from gensim.models import KeyedVectors
from models.nlp.conv_text_adv import (
    Conv1dTextAdvClassifier,
    Conv2dTextAdvClassifier
)

def pick_model(cfg: CommonCfgParams, load_path: str = None):
    """ 根据 model_name 返回对应的模型实例
    """
    model = None
    if cfg.model == 'lstm':
        model = LSTMTextClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            bidirectional=True, output_dim=cfg.output_dim, layers=cfg.L, 
            atten_key=cfg.atten
        )
    elif cfg.model == 'lstm-adv':
        model = LSTMTextAdvClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            bidirectional=True, output_dim=cfg.output_dim, layers=cfg.L,
            atten_key=cfg.atten
        )
    elif cfg.model == 'conv1d':
        model = Conv1dTextClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim, atten_key=cfg.atten
        )
    elif cfg.model == 'conv2d':
        model = Conv2dTextClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim, max_len=cfg.max_len, atten_key=cfg.atten
        )
    elif cfg.model == 'conv1d-adv':
        model = Conv1dTextAdvClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim, atten_key=cfg.atten
        )
    elif cfg.model == 'conv2d-adv':
        model = Conv2dTextAdvClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim,
            output_dim=cfg.output_dim, atten_key=cfg.atten
        )
    elif cfg.model == 'tcn':
        model = TCNTextClassifier(
            vocab_size=cfg.vocab_size, embedding_dim=cfg.embedding_dim, tcn_channels=[128, 128, 128], 
            output_dim=cfg.output_dim
        )
    
    
    if model is None:
        raise ValueError(f"Unknown model name: {cfg.model}")
    
    if load_path is not None: 
        model = io.load_model_weights(model, load_path, device=cfg.device)
    
    return model


def pick_embedding_encoder(cfg: NlpCfgParams, load_path: str = None, vocab = None):
    """ 根据 encoder-name 返回对应的 embedding encoder 模型,
        需要注意，encoder 是不参与训练的，因此在返回之后必须冻结其参数
    """
    encoder = None

    # 使用预训练模型原先的 embedding 模块来做转化
    if cfg.encoder == "default": 
        model = pick_model(cfg, load_path)
        encoder = model.embedding

        # 冻结参数：不参与梯度更新
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder
    
    if  cfg.encoder == "word2vec":
        model = KeyedVectors.load(load_path, mmap='r')

        def m(x):
            if isinstance(x, torch.Tensor):
                x = x.tolist()
            index2word = {index: word for word, index in vocab.items()}
            # 输入的是批量数据，每句话单独处理
            embeddings = []
            for seq in x:
                words = [index2word.get(idx, '<UNK>') for idx in seq]
                vectors = [model[word] if word in model else np.zeros(model.vector_size) for word in words]
                embeddings.append(vectors)
            return torch.tensor(np.array(embeddings), dtype=torch.float32) 
        
        encoder = m
        return encoder
    
    if cfg.encoder == "fasttext":
        model = KeyedVectors.load(load_path, mmap='r')
        
        def m(x):
            if isinstance(x, torch.Tensor):
                x = x.tolist()
            index2word = {index: word for word, index in vocab.items()}
            # 输入的是批量数据，每句话单独处理
            embeddings = []
            for seq in x:
                words = [index2word.get(idx, '<UNK>') for idx in seq]
                vectors = [model[word] if word in model else np.zeros(model.vector_size) for word in words]
                embeddings.append(vectors)
            return torch.tensor(np.array(embeddings), dtype=torch.float32) 
        
        encoder = m
        return encoder


    # 此时会用恒等映射模块来做 encoder，相当于跳过了原始模型的 embedding 模块
    if cfg.encoder == "id":
        return nn.Identity()
    

    return encoder