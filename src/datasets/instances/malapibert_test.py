from src.datasets.instances.malapibert import MalAPIBertDataset
from transformers import AutoTokenizer
from typing import Dict
import torch
import pytest


codebert_tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path = 'microsoft/codebert-base', 
    add_prefix_space=True
)

class TestMalAPIBertDataset:
    """ NOTE 如果因为网络问题, 无法访问HF官方网站, 可以先执行命令: export HF_ENDPOINT=https://hf-mirror.com 
        或者这个命令加到终端的配置文件之中 (~/.bashrc 或 ~/.zshrc)
    """
    
    # 准备测试数据
    raw_text = "ldrloaddll ldrgetprocedureaddress ldrloaddll ldrgetprocedureaddress"
    raw_label_str = "Trojan"
    
    # 模拟 label mapping: Trojan -> 1
    label_idx = 1
    
    @pytest.fixture
    def dataset(self):
        """初始化一个较小 max_len 的数据集用于快速测试"""
        return MalAPIBertDataset(
            texts=[self.raw_text], 
            labels=[self.label_idx],
            bert_tokenizer = codebert_tokenizer,
            max_len=16 , 
            is_split_into_words=False, 
            text_pipeline = lambda x: x
        )
        
        
    @pytest.fixture
    def splited_text_dataset(self):
        """初始化一个较小 max_len 的数据集用于快速测试"""
        return MalAPIBertDataset(
            texts=[self.raw_text], 
            labels=[self.label_idx],
            bert_tokenizer = codebert_tokenizer,
            max_len=16,
        )
        
    def test_len(self, dataset):
        """测试数据集大小"""
        assert len(dataset) == 1
    
    def _check_one_sample(self, sample: Dict):
        # 检查键是否存在
        assert 'input_ids' in sample and 'attention_mask' in sample and 'labels' in sample
        
        # 检查类型
        assert isinstance(sample['input_ids'], torch.Tensor)
        assert isinstance(sample['labels'], torch.Tensor)
        
        # 检查维度 (flatten 后的效果)
        # input_ids 应该是 [max_len] 而不是 [1, max_len]
        assert sample['input_ids'].shape == (16,) 
        assert sample['attention_mask'].shape == (16,)
        
        # labels 应该是标量 (0-d tensor)
        assert sample['labels'].ndim == 0 
        assert sample['labels'].item() == self.label_idx
        
    
    def test_splited_output_structure(self, splited_text_dataset):
        """测试输出字典的结构和维度"""
        sample = splited_text_dataset[0]
        self._check_one_sample(sample)
        
        

    def test_output_structure(self, dataset):
        """测试输出字典的结构和维度"""
        sample = dataset[0]
        self._check_one_sample(sample)
        
    
    def test_special_tokens(self, dataset):
        """测试 [CLS] 和 [SEP] 是否正确添加"""
        sample = dataset[0]
        input_ids = sample['input_ids']
        tokenizer = dataset.tokenizer
        
        # BERT 的 [CLS] id 通常是 101, [SEP] 是 102
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        
        # 句首必须是 CLS
        assert input_ids[0] == cls_id
        
        # 句中必须包含 SEP (因为输入文本很短，肯定在 16 以内)
        assert sep_id in input_ids
        
        # SEP 之后应该是 PAD
        sep_index = (input_ids == sep_id).nonzero(as_tuple=True)[0][0]
        if sep_index < len(input_ids) - 1:
            assert input_ids[sep_index + 1] == pad_id

    def test_attention_mask(self, dataset):
        """测试 mask 是否正确 (内容等于1，padding等于0)"""
        sample = dataset[0]
        input_ids, mask, labels = (
            sample["input_ids"], 
            sample["attention_mask"], 
            sample["labels"]
        )
        
        print(f"input_ids = {input_ids}")
        print(f"labels = {labels}")
        print(f"mask = {mask}")
        
        tokenizer = dataset.tokenizer
        
        # 找到 PAD 的位置
        is_padding = (input_ids == tokenizer.pad_token_id)
        
        # Padding 位置 mask 应为 0
        assert torch.all(mask[is_padding] == 0)
        
        # 对于填空位置, i.e Padding 位置 mask 应w为 1
        assert torch.all(mask[~is_padding] == 1)

    def test_truncation(self):
        """单独测试截断功能"""
        # 构造一个超长的输入
        long_text = "ldrloaddll " * 100
        dataset = MalAPIBertDataset(
            texts=[long_text], 
            labels=[0], 
            max_len=10,
            bert_tokenizer = codebert_tokenizer,
            is_split_into_words=False,
            text_pipeline = lambda x: x
        )
        
        sample = dataset[0]
        assert sample['input_ids'].shape == (10,)
        # 确保最后一个 token 并非 pad (说明被填满了/截断了)
        
        # 注意：截断后最后一个 token 应为 [SEP]
        tokenizer = dataset.tokenizer
        assert sample['input_ids'][-1] == tokenizer.sep_token_id