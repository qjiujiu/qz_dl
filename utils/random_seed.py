import torch
import random
import numpy as np

def set_random_seed(seed: int):
    """ 固定随机数种子 """
    random.seed(seed)  # Python 随机数种子
    np.random.seed(seed)  # Numpy 随机数种子
    torch.manual_seed(seed)  # PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果有多个 GPU 使用
    torch.backends.cudnn.deterministic = True  # 确保使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭 cudnn.benchmark，这对于每次输入相同大小时比较好
