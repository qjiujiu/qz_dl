import argparse
import torch
import random
import numpy as np

from dataclasses import fields, is_dataclass
from config.params_parser.params_template import (
    CommonCfgParams,
    CvCfgParams,
    NlpCfgParams,
    AdvCfgParams 
)


def fixture_random_seed(seed: int):
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



class ArgsParser:
    def __init__(self):
        self.args = self._parse_args()
        self.args_dict = vars(self.args)

        fixture_random_seed(self.args.seed)

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="深度学习项目通用参数解析")

        # --- 公共训练参数 ---
        parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="批处理大小")
        parser.add_argument("--dropout-prob", "-drop", type=float, default=0.5, help="随机失活的概率")
        parser.add_argument("--epochs", "-ep", type=int, default=10, help="训练轮数")
        parser.add_argument("--lr", type=float, default=0.01, help="学习率")
        parser.add_argument("--device", type=str, default="cuda", help="运行设备，默认 cuda，若不可用则自动转为 cpu 模式")
        parser.add_argument("--dataset", type=str, default=None, help="使用的数据集名称")
        parser.add_argument("--seed",  type=int, default=3407, help="随机数种子")
        parser.add_argument("--model", "-m", type=str, default=None, help="使用的模型的名称")
        parser.add_argument("--checkpoint-path", "-cp", type=str, help="模型保存或载入路径")
        parser.add_argument("--load-path", "-lp", type=str, default=None, help="模型载入路径，亦或是其它用途的路径")
        
        # --- 预留超参数 ---
        parser.add_argument("--alpha", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--beta", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--gama", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--n", type=int, default=1, help="预留超参数")
        parser.add_argument("--L", type=int, default=0, help="预留超参数")
        parser.add_argument("--t", type=int, default=1, help="预留超参数")
        parser.add_argument("--x", "-x", type=int, default=1, help="预留超参数")
        
        # 添加对抗攻击相关的超参数
        parser.add_argument("--fgsm_epsilon", type=float, default=0.1, help="FGSM 扰动强度")
        parser.add_argument("--pgd_epsilon", type=float, default=0.1, help="PGD 最大扰动范围")
        parser.add_argument("--pgd_alpha", type=float, default=0.01, help="PGD 每步更新幅度")
        parser.add_argument("--pgd_iters", type=int, default=5, help="PGD 迭代次数")
        parser.add_argument("--adv-type", type=str, default="fgsm", help="对抗攻击模式")

        # --- NLP 专用参数 ---
        parser.add_argument("--vocab-size", "-vs", type=int, help="词表大小")
        parser.add_argument("--embedding-dim", "-eb", type=int, help="词向量维度")
        parser.add_argument("--hidden-dim", type=int, help="隐藏层维度")
        parser.add_argument("--output-dim", type=int, help="输出类别数")
        parser.add_argument("--max-len",  type=int, help="最大序列长度")
        parser.add_argument("--only-embed", action="store_true", help="是否直接传入 embedding 向量进行训练")
        parser.add_argument("--encoder", "-ec", type=str, default=None, help="使用何种编码器来将文本转为 embedding")


        # --- 其它场景的参数 ---
        parser.add_argument("--max-feature", type=int, default=1000, help="机器模型特征提取器最大允许提取特征数")
        parser.add_argument("--n-gram", type=int, default= 1, help="机器模型 n-gram 特征提取")

        return parser.parse_args()
    

    @staticmethod
    def create_config(config_class, **kwargs) -> CommonCfgParams:
        """ 通过 kwargs 构造 config_class 实例，忽略未定义的字段
        """
        if not is_dataclass(config_class):
            raise TypeError("config_class 必须是一个 dataclass")

        
        valid_fields = set()
        for f in fields(config_class):
            if f.init:
                valid_fields.add(f.name)
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return config_class(**filtered)

    def create_cv_config(self) -> CvCfgParams:
        return self.create_config(CvCfgParams, **self.args_dict)

    def create_nlp_config(self) -> NlpCfgParams:
        return self.create_config(NlpCfgParams, **self.args_dict)
    
    def create_adv_config(self) -> AdvCfgParams:
        return self.create_config(AdvCfgParams, **self.args_dict)




    