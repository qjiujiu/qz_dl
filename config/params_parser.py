from dataclasses import dataclass, asdict
from typing import Optional

# 超参数配置类
from dataclasses import dataclass, field
import torch



@dataclass
class CommonCfgParams:
    batch_size: int = 32                   # 批处理大小
    epochs: int = 10                       # 训练轮数
    lr: float = 0.01                       # 学习率
    dropout_prob: float = 0.5              # 随机失活概率
    device: str = "cuda"                   # 运行设备
    checkpoint_path: Optional[str] = None  # 模型保存路径，使用 Optional 和 None 表示可选字段
    
    def __post_init__(self):
        # 自动根据设备环境选择
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def as_dict(self):
        return asdict(self)


@dataclass
class CvCfgParams(CommonCfgParams):
    pass


@dataclass
class NlpCfgParams(CommonCfgParams):
    vocab_size: Optional[int] = None             # 词表大小
    embedding_dim: Optional[int] = None          # 词向量维度
    hidden_dim: Optional[int] = None             # 隐藏层维度
    output_dim: Optional[int] = None             # 输出类别数
    max_len: Optional[int] = None                # 最大序列长度
  
    only_embed: bool = False                     # 直接传入embedding 向量进行训练，而不传入文件索引
    
    def __post_init__(self):
        pass
