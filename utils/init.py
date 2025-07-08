import torch
import random
import numpy as np

   
def random_seed(seed: int):
    """ 固定随机数种子 """
    # Python 随机数种子
    random.seed(seed)
    
    # Numpy 随机数种子
    np.random.seed(seed)

    # PyTorch CPU/GPU 随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    # 确保使用确定性算法，并且关闭 cudnn.benchmark，这对于每次输入相同大小时比较好
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  