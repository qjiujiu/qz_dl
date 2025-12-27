from src.datasets.instances.malanalysis import MalAnalysisDataset, build_datamodule

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch



class TestMalAnalysisDataset:
    """MalAnalysisDataset 数据集类的单元测试"""

    @pytest.fixture
    def dummy_csv_path(self, tmp_path):
        """创建一个临时的 CSV 文件用于测试"""
        # 构造模拟数据: 
        # 3个良性(0), 2个恶意(1), 序列长度设为 5
        data = {
            'hash': ['h1', 'h2', 'h3', 'h4', 'h5'],
            't_0': [1, 2, 3, 4, 5],
            't_1': [6, 7, 8, 9, 0],
            't_2': [1, 2, 3, 4, 5],
            't_3': [6, 7, 8, 9, 0],
            't_4': [1, 2, 3, 4, 5],
            'malware': [0, 1, 0, 1, 0] # 标签
        }
        df = pd.DataFrame(data)
        
        # 保存到临时目录
        file_path = tmp_path / "dynamic_api_call_seq100.csv"
        df.to_csv(file_path, index=False)
        return file_path

    @pytest.fixture
    def mock_context(self, dummy_csv_path):
        """模拟 ExpContext 配置对象"""
        # 模拟 DataConfig
        mock_data_config = MagicMock()
        mock_data_config.data_dir = dummy_csv_path.parent
        mock_data_config.test_size = 0.4  # 验证集占 40% (五个总样本里面分出两个)
        mock_data_config.num_workers = 0  # 测试时不使用多进程
        mock_data_config.pin_memory = False

        # 模拟 TrainConfig
        mock_train_config = MagicMock()
        mock_train_config.batch_size = 2
        mock_train_config.seed = 42

        # 模拟 ExpContext
        mock_ctx = MagicMock()
        mock_ctx.data_config = mock_data_config
        mock_ctx.train_config = mock_train_config
        
        return mock_ctx

    def test_dataset_initialization(self, dummy_csv_path):
        """测试数据集加载逻辑"""
        dataset = MalAnalysisDataset(csv_path=dummy_csv_path)

        # 检查样本总数
        assert len(dataset) == 5
        
        # 检查特征长度 (过滤掉了 hash 和 malware，剩下 5 列 t_0~t_4)
        assert dataset.seq_len == 5
        assert dataset.features.shape == (5, 5)
        assert dataset.features.dtype == torch.long

        # 检查标签
        assert dataset.labels.shape == (5,)
        assert dataset.labels.dtype == torch.long
        # 检查是否正确读取了 [0, 1, 0, 1, 0]
        assert torch.equal(dataset.labels, torch.tensor([0, 1, 0, 1, 0], dtype=torch.long))

    def test_getitem(self, dummy_csv_path):
        """测试返回的数据格式"""
        dataset = MalAnalysisDataset(csv_path=dummy_csv_path)
        
        # 获取第0个样本
        features, label = dataset[0]
        
        # 检查 features
        assert isinstance(features, torch.Tensor)
        assert features.shape == (5,) # seq_len
        # 对应 CSV 第一行数据 [1, 6, 1, 6, 1]
        
        # 检查 label
        assert isinstance(label, torch.Tensor)
        assert label.ndim == 0 # scalar
        assert label.item() == 0

    def test_file_not_found(self):
        """测试文件不存在时是否报错"""
        with pytest.raises(FileNotFoundError):
            MalAnalysisDataset(csv_path="non_existent_file.csv")

    def test_build_datamodule(self,  mock_context):
        """ 测试 build_datamodule 的切分和 Loader 构建
            使用 @patch 装饰器 mock 掉 _print_data_distribution 防止测试时报错
        """
        train_loader, val_loader = build_datamodule(mock_context)

        # 检查数据集切分大小
        # total=5, test_size=0.4 => val=2, train=3
        assert len(train_loader.dataset) == 3
        assert len(val_loader.dataset) == 2

        # 检查 DataLoader 是否可迭代
        batch_x, batch_y = next(iter(train_loader))
        
        assert batch_x.shape[0] <= 2 
        assert batch_y.shape[0] <= 2
        
        # 检查数据维度 [batch, seq_len]
        assert batch_x.shape[1] == 5