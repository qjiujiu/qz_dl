from src.utils.logx import logger
from src.utils.text_preprocessing import (
    identity_preprocess,
    dedup_preprocess, 
    default_preprocess,
)
from src.utils.dumps import read_pickle, write_pickle
from src.schemas.context import ExpContext, DataConfig
from src.datasets.instances.malapi2019 import (
    _load_raw_text_data, 
    _print_data_distribution, 
)

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Any, Callable, Union
from pathlib import Path
import torch
import pickle



""" NOTE 
BERT (Bidirectional Encoder Representations from Transformers), 通过海量问题进行预训练, 其训练任务包含两个
    - 完全填空   Masked LM 
    - 下一句预测  Next Sentence Predict 

    其中包含两个特殊符号: 
    
    [CLS] Classification Token: 永远在输入序列的首个位置, 其作用是汇总整个句子的语义信息, 经过十二层BERT处理之后, CLS 
          即为整个句子的信息 embedding 
          
    [SEP] Separator Token: 放在句子的结尾，或者两个句子中间, 告诉模型这里是一句话的结束，哪怕只有一个句子也要在末尾加上这个
          结束符, 假设 API 调用序列等于 Open Read Close, 进入 Bert 之前要先变成 [CLS] Open Read Close [SEP] 
    
    attention mask 相当于一个掩码, 标记没有实际意义的 padding 数据
"""



# 确保 Tokenizer 开启 add_prefix_space 以便支持 is_split_into_words=True
_codebert_tokenizer_creator = lambda: AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path = 'microsoft/codebert-base', 
    add_prefix_space = True 
)

# 改造 Bert 模型
class MalAPIBertDataset(Dataset):
    def __init__(self, 
            texts: List[str], 
            labels: List[int], 
            bert_tokenizer: Any, 
            max_len: int = 200, 
            text_pipeline: Callable[[str], List[str]] = identity_preprocess,
            is_split_into_words=False,
            **kwargs
        ):
        """ 利用预训练的 BERT 模型进行Emedding
        :param texts: API 序列列表，例如 ["CreateFile ReadFile CloseHandle", ...], API 之间通过空格隔开
        :param labels: 标签列表
        :param  bert_tokenizer 分词器, 通过预训练模型名称进行创建, 推荐: 
                - 'bert-base-uncased', 使用了英文维基百科 + 图书语料库
                - 'microsoft/codebert-base', 使用了 GitHub 海量代码预训练的模型, 包含 Python, Java, C++, Go, etc 六种语言及其注释
        :param max_len: 截断长度
        :param text_pipeline: 预处理函数
        """
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
           
        self.is_split_into_words = is_split_into_words
        self.text_pipeline = text_pipeline
        
        
        # 加载 BERT 的分词器
        self.tokenizer = bert_tokenizer
        self.vocab_size = len(self.tokenizer)
       

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Dict:
        text = self.texts[idx]
        label = self.labels[idx]

        
        # 如果设置了 is_split_into_words 类型则为 List[str], 否则应为 str
        tokens_or_text: Union[str, List[str]] = self.text_pipeline(text)
        
        # 使用 BERT 分词器进行编码
        # 设置 is_split_into_words=True     如果传入 List[str] 则要设置, 否则的默认 False (i.e. 传入的 text 是一个 str)
        # 设置 add_special_tokens=True:     自动在句首加 [CLS]，句尾加 [SEP]
        # 设置 padding='max_length':        自动填充到 max_len
        # 设置 truncation=True:             自动截断
        # 设置 return_attention_mask=True:  返回 attention_mask
        
        encoding = self.tokenizer.encode_plus(
            tokens_or_text,
            is_split_into_words=self.is_split_into_words,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False, 
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            # 返回 PyTorch 张量
            return_tensors='pt',
        )
        
        return {
            # 需要 flatten 是因为 tokenizer 返回  [1, max_len]，我们需要 [max_len]
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        


def _load_data(cfg: DataConfig) -> Tuple[Dataset, Dataset, Any]:
    """根据配置加载 MalAPI 数据集 (BERT版)"""
    data_dir: Path = Path(cfg.data_dir)
    cache_dir: Path = data_dir / "preprocessed" 
    cache_dir.mkdir(parents=True, exist_ok=True) 
    
    # 定义缓存文件路径 (不需要 vocab.pkl 了)
    cache_train_path = cache_dir / f"{cfg.dataset_name}-train.pkl"
    cache_test_path = cache_dir / f"{cfg.dataset_name}-test.pkl"
    
    # 加载数据 (Cache 或 Raw)
    if cache_train_path.exists() and cache_test_path.exists():
        logger.debug(f"Cache Hit! Loading dataset: {cfg.dataset_name}")        
        train_data, test_data = read_pickle(cache_train_path, cache_test_path)
        
    else:
        logger.debug("Cache miss. Processing raw text data...")
        X_train, X_test, y_train, y_test = _load_raw_text_data(data_dir)
        
        train_data, test_data = (X_train, y_train), (X_test, y_test)
        write_pickle(
            (cache_train_path, train_data),
            (cache_test_path, test_data),
        )
        
        logger.debug("Cache saved successfully.")

    # 打印分布
    all_labels = train_data[1] + test_data[1]
    _print_data_distribution(all_labels, title="Full Dataset")
    
    if not cfg.nlp_config or not cfg.nlp_config.max_len:
         raise ValueError("Config Error: 'nlp_config.max_len' is required.")
    
    # 初始化 Tokenizer
    max_len = cfg.nlp_config.max_len
    tokenizer = _codebert_tokenizer_creator()
    
    pipeline_map = {
        "malapi2019": {"is_split_into_words": False, "text_pipeline": identity_preprocess},
        "malapi2019-gc": {"is_split_into_words": True, "text_pipeline": dedup_preprocess},
    }
    
    _pipeline = pipeline_map.get(cfg.dataset_name, default_preprocess)
    
    
    if _pipeline is None:
        valid_names = list(pipeline_map.keys())
        raise ValueError(
            f"Config Error: Dataset name '{cfg.dataset_name}' is invalid/unsupported.\n"
            f"Supported datasets: {valid_names}"
        )
        

    # 实例化 Dataset
    train_ds = MalAPIBertDataset(train_data[0], train_data[1], tokenizer, max_len=max_len, **_pipeline)
    test_ds = MalAPIBertDataset(test_data[0], test_data[1],  tokenizer, max_len=max_len,  **_pipeline)
    
    return train_ds, test_ds, tokenizer



def build_datamodule(ctx: ExpContext) -> Tuple[DataLoader, DataLoader]:
    """ 外部调用的唯一入口: 直接调用模块函数加载
        返回训练数据加载器、验证数据的加载器、词表大小
    """
    cfg = ctx.data_config
    train_ds, test_ds, tokenizer = _load_data(cfg)
    
    vocab_size = len(tokenizer)
    cfg.nlp_config.vocab_size = vocab_size  
    logger.debug(f"Auto-updating vocab_size to {vocab_size}")
    
        
    # 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=ctx.train_config.batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=ctx.train_config.batch_size)
    
    return train_loader, val_loader
