import argparse
from dataclasses import fields, is_dataclass
from config.params_parser.params_template import (
    CommonCfgParams,
    CvCfgParams,
    NlpCfgParams 
)


class ArgsParser:
    def __init__(self):
        self.args = self._parse_args()
        self.args_dict = vars(self.args)

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="深度学习项目通用参数解析")

        # --- 公共训练参数 ---
        parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="批处理大小")
        parser.add_argument("--dropout-prob", "-drop", type=float, default=0.5, help="随机失活的概率")
        parser.add_argument("--epochs", "-ep", type=int, default=10, help="训练轮数")
        parser.add_argument("--lr", type=float, default=0.01, help="学习率")
        parser.add_argument("--checkpoint-path", type=str, help="模型保存路径")
        parser.add_argument("--device", type=str, default="cuda", help="运行设备，默认 cuda，若不可用则自动转为 cpu 模式")
        parser.add_argument("--dataset", type=str, default=None, help="使用的数据集名称")
        parser.add_argument("--model", type=str, default=None, help="使用的模型的名称")
        

        # --- 预留超参数 ---
        parser.add_argument("--alpha", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--beta", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--gama", type=float, default=0.5, help="预留超参数")
        parser.add_argument("--n", type=int, default=1, help="预留超参数")
        parser.add_argument("--L", type=int, default=0, help="预留超参数")
        parser.add_argument("--t", type=int, default=1, help="预留超参数")
        parser.add_argument("--x", type=int, default=1, help="预留超参数")

        # --- NLP 专用参数 ---
        parser.add_argument("--vocab-size", type=int, help="词表大小")
        parser.add_argument("--embedding-dim", type=int, help="词向量维度")
        parser.add_argument("--hidden-dim", type=int, help="隐藏层维度")
        parser.add_argument("--output-dim", type=int, help="输出类别数")
        parser.add_argument("--max-len", type=int, help="最大序列长度")
        parser.add_argument("--only-embed", action="store_true", help="是否直接传入 embedding 向量进行训练")

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




    