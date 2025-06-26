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

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="深度学习项目通用参数解析")

        # --- 公共训练参数 ---
        parser.add_argument("--batch-size", "-bs", type=int, default=1024, help="批处理大小")
        parser.add_argument("--dropout-prob", "-drop", type=float, default=0.5, help="随机失活的概率")
        parser.add_argument("--epochs", "-ep", type=int, default=10, help="训练轮数")
        parser.add_argument("--lr", type=float, default=0.01, help="学习率")
        parser.add_argument("--checkpoint-path", type=str, help="模型保存路径")
        parser.add_argument("--device", type=str, default="cuda", help="运行设备，默认 cuda，若不可用则自动转为 cpu 模式")

        # --- NLP 专用参数 ---
        parser.add_argument("--vocab-size", type=int, help="词表大小")
        parser.add_argument("--embedding-dim", type=int, help="词向量维度")
        parser.add_argument("--hidden-dim", type=int, help="隐藏层维度")
        parser.add_argument("--output-dim", type=int, help="输出类别数")
        parser.add_argument("--max-len", type=int, help="最大序列长度")
        parser.add_argument("--only-embed", action="store_true", help="是否直接传入 embedding 向量进行训练")

        args = parser.parse_args()
        return vars(args)

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
        return self.create_config(CvCfgParams, **self.args)

    def create_nlp_config(self) -> NlpCfgParams:
        return self.create_config(NlpCfgParams, **self.args)




    