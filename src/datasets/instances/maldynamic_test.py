import pytest
import torch
import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# 假设你的文件路径是 src/datasets/instances/maldynamic.py
from src.datasets.instances.maldynamic import MalDynamic, build_datamodule

class TestMalDynamicDataset:
    """测试 MalDynamic Dataset 类的核心逻辑"""

    @pytest.fixture
    def sample_data(self):
        vocab = {"<unk>": 0, "<pad>": 1, "create": 2, "delete": 3, "open": 4}
        apis = [["create", "open"], ["open", "unknown_api", "delete"]]
        # 构建两个多标签向量, 补齐 15-dim
        labels = [[0, 1, 0] + [0]*12, 
                  [1, 0, 1] + [0]*12]
        return apis, labels, vocab

    def test_getitem_padding(self, sample_data):
        """测试填充功能 (Pad)"""
        apis, labels, vocab = sample_data
        max_len = 5
        dataset = MalDynamic(apis, labels, vocab, max_len=max_len)
        
        x, y = dataset[0] # ["create", "open"]
        print(x)
        print(y)
        
        assert x.shape == (5,)
        expected_x = torch.tensor([2, 4, 1, 1, 1], dtype=torch.long)
        
        assert torch.equal(x, expected_x)
        assert y.dtype == torch.float 

    def test_getitem_truncation(self, sample_data):
        """测试截断功能 (Truncate)"""

        apis, labels, vocab = sample_data
        max_len = 2 
        dataset = MalDynamic(apis, labels, vocab, max_len=max_len)
        
        
        # ["open", "unknown_api", "delete"], 应为 [open(4), unk(0)]，delete 被截断
        x, y = dataset[1] 
        assert x.shape == (2,)
        
        expected_x = torch.tensor([4, 0], dtype=torch.long)
        assert torch.equal(x, expected_x)
        

    def test_unknown_token(self, sample_data):
        """测试未知 Token 处理"""
        apis, labels, vocab = sample_data
        dataset = MalDynamic(apis, labels, vocab, max_len=5)
        
        # ["open", "unknown_api", "delete"], 其中 unknown_api 应该变成 0
        x, _ = dataset[1] 

        assert x[1] == 0


class TestBuildDataModule:
    """测试 build_datamodule 的构建与词表缓存逻辑"""

    @pytest.fixture
    def mock_ctx(self, tmp_path):
        """模拟 Context 对象"""
        ctx = MagicMock()
        ctx.data_config.data_dir = tmp_path
        ctx.data_config.min_freq = 1
        ctx.data_config.max_len = 10
        ctx.data_config.test_size = 0.5
        ctx.data_config.num_workers = 0
        ctx.data_config.pin_memory = False
        
        ctx.train_config.seed = 42
        ctx.train_config.batch_size = 2
        
        ctx.network_config = MagicMock() # 用于回填 vocab_size
        return ctx

    @pytest.fixture
    def mock_data_content(self):
        """模拟数据文件内容"""
        return json.dumps({
            "apis": [["api_a", "api_b"], ["api_c"]],
            "labels": [[0]*15, [1]*15]
        })

    def test_load_existing_vocab(self, mock_ctx, mock_data_content):
        """场景 A: vocab.json 存在，应该直接读取，不调用 build_vocab"""
        
        # 模拟 vocab 文件内容
        vocab_content = json.dumps({"<unk>": 0, "<pad>": 1, "api_a": 2})
        
        # 我们需要模拟两个文件的打开：一个是数据文件，一个是词表文件
        def side_effect(filename, *args, **kwargs):
            fname = str(filename)
            if "vocab.json" in fname:
                return mock_open(read_data=vocab_content).return_value
            elif "json" in fname: # 其他数据文件
                return mock_open(read_data=mock_data_content).return_value
            return MagicMock()

        # Mock Path.exists
        with patch("pathlib.Path.exists") as mock_exists, \
             patch("builtins.open", side_effect=side_effect) as mock_file, \
             patch("src.datasets.instances.maldynamic.build_vocab") as mock_build_vocab:
            
            # 设定：所有文件都存在
            mock_exists.return_value = True
            
            # 执行
            train_dl, val_dl = build_datamodule(mock_ctx)
            
            # 断言：build_vocab 没有被调用
            mock_build_vocab.assert_not_called()
            
            # 断言：ctx 中的 vocab_size 被正确更新 (unk, pad, api_a 共3个)
            assert mock_ctx.network_config.vocab_size == 3
            
            # 简单的 DataLoader 检查
            assert isinstance(train_dl, torch.utils.data.DataLoader)

    def test_build_and_save_vocab_if_missing(self, mock_ctx, mock_data_content):
        """场景 B: vocab.json 不存在，应该调用 build_vocab 并保存"""
        
        generated_vocab = {"<unk>": 0, "<pad>": 1, "api_gen": 2}
        
        # 这里的逻辑稍微复杂一点：数据文件存在，但 vocab 不存在
        def exists_side_effect(path):
            if "vocab.json" in str(path):
                return False
            return True # 数据文件存在

        # 模拟 open: 读数据文件时返回内容，写 vocab 时返回一个 mock 对象
        file_mock = mock_open(read_data=mock_data_content)
        
        with patch("pathlib.Path.exists", side_effect=exists_side_effect, autospec=True), \
             patch("builtins.open", file_mock), \
             patch("src.datasets.instances.maldynamic.build_vocab") as mock_build_vocab, \
             patch("json.dump") as mock_json_dump:
                 
            # 设定 build_vocab 的返回值
            mock_build_vocab.return_value = generated_vocab
            
            # 执行
            build_datamodule(mock_ctx)
            
            # 断言：build_vocab 被调用了
            mock_build_vocab.assert_called_once()
            
            # 断言：json.dump 被调用了 (说明执行了保存操作)
            mock_json_dump.assert_called()
            # 检查保存的内容是否是生成的 vocab
            args, _ = mock_json_dump.call_args
            assert args[0] == generated_vocab
            
            # 断言：config 更新
            assert mock_ctx.network_config.vocab_size == 3