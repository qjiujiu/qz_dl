from dataclasses import dataclass, asdict
from typing import Optional

# 超参数配置类
from dataclasses import dataclass, field
import torch



@dataclass
class CommonCfgParams:
    batch_size: int = 8                    # 批处理大小
    epochs: int = 30                       # 训练轮数
    lr: float = 0.01                       # 学习率
    dropout_prob: float = 0.5              # 随机失活概率
    device: str = "cuda"                   # 运行设备
    dataset: Optional[str] = None          # 使用的数据集的名称
    model: Optional[str] = None            # 使用的模型名称
    load_path: Optional[str] = None        # encoder模型权重载入路径
    checkpoint_path: Optional[str] = None  # 模型保存路径
    seed: int = 3407                       # 随机数种子

    # 预留的超参数，相当于提前占用了这些字母，这些参数可能用于任何地方
    alpha: float = 0.5
    beta: float = 0.5
    gama: float = 0.5
    n: int = 1
    L: int = 0           # mlp 隐藏层层数
    t: int = 1
    x: int = 1
    
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
    
  
    only_embed: bool = False                     # 直接传入 embedding 向量进行训练，而不传入文件索引
    encoder: str = None                          # 使用何种编码器来将文本输入转为 embedding 向量输入
    atten: Optional[str] = None                  # 使用何种注意力机制模块
    
    def __post_init__(self):
        pass

@dataclass
class AdvCfgParams(NlpCfgParams):
    fgsm_epsilon: Optional[int] = 0.1           # FGSM 扰动强度
    pgd_epsilon: Optional[int] = 0.1            # PGD 最大扰动范围
    pgd_alpha: Optional[int] = 0.01             # PGD 每步更新幅度
    pgd_iters: Optional[int] = 5                # PGD 迭代次数
    adv_type: Optional[str] = "fgsm"            # 对抗攻击模式 

    def __post_init__(self):
        pass
