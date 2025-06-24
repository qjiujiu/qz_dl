from dataclasses import dataclass, asdict
from typing import Optional

# 超参数配置类
from dataclasses import dataclass, field
import torch

@dataclass
class CommonCfgParams:
    batch_size: int             # 批处理大小
    epochs: int                 # 训练轮数
    lr: float                   # 学习率
    device: str                 # 运行设备
    checkpoint_path: str        # 模型保存路径
    
    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
   


@dataclass
class CvCfgParams(CommonCfgParams):
    pass

@dataclass
class NlpCfgParams(CommonCfgParams):
    vocab_size: int             # 词表大小
    embedding_dim: int          # 词向量维度
    hidden_dim: int             # 隐藏层维度
    output_dim: int             # 输出类别数
    max_len: int                # 最大序列长度
   
  
    only_embed: bool = False    # 直接传入embedding 向量进行训练，而不传入文件索引
    

    def __post_init__(self):
        pass