
from src.schemas.base_enums import TaskType
from src.schemas.context import DataConfig, NLPConfig, TrainConfig
from src.datasets.instances.malapi2019 import (
    MalAPITextDataset,
    _load_raw_text_data,
    _load_data,
    build_datamodule,
    build_vocab,
)


from unittest.mock import patch, MagicMock
from pathlib import Path
import torch

import pytest
import tempfile




@pytest.fixture
def mock_raw_data_dir():
    """创建临时目录并写入 mock 原始数据"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "malapi2019"
        data_dir.mkdir(parents=True)

        # 模拟 raw 数据, 10 samples
        texts = [
            "CreateFile open registry key",
            "URLDownloadToFile connect InternetOpen",
            "VirtualAlloc inject shellcode",
            "RegSetValue malware persistence",
            "CreateProcess spawn cmd.exe"
        ] * 2 

        labels = ["Trojan", "Downloader", "Backdoor", "Spyware", "Worms"] * 2

        # 写入文件
        (data_dir / "all_analysis_data.txt").write_text("\n".join(texts), encoding="utf-8")
        (data_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

        yield data_dir


@pytest.fixture
def mock_data_config(mock_raw_data_dir):
    return DataConfig(
        dataset_name="malaapi1209",
        data_dir = str(mock_raw_data_dir),
        task_type = TaskType.NLP,
        nlp_config = NLPConfig(
            min_freq = 1,
            max_len = 16 
        )
    )


@pytest.fixture
def mock_exp_context(mock_data_config):
    # 简易 mock ExpContext（按你实际结构调整）
    ctx = MagicMock()
    ctx.data_config = mock_data_config
    ctx.train_config = TrainConfig(batch_size=4)
    return ctx


def test_load_raw_text_data(mock_raw_data_dir):
    X_train, X_test, y_train, y_test = _load_raw_text_data(mock_raw_data_dir, test_size=0.2)
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert all(isinstance(x, str) for x in X_train + X_test)
    assert all(isinstance(y, int) for y in y_train + y_test)


def test_dataset_getitem():
    # 构建小词表
    texts = [
        "hello world", "foo bar baz"
    ]
    vocab = build_vocab(texts, min_freq=1)
    
    print(f"texts: {texts}")
    print(f"vocab: {vocab}")
    
    ds = MalAPITextDataset(
        texts=["hello foo", "bar"], 
        labels=[0, 1], 
        vocab=vocab, 
        max_len=5
    )
    indices, label = ds[0]
    print(f"shape = {indices.shape}, label: {indices}")
    print(f"shape = {label.shape}, label: {label}")
    
    assert indices.shape == (5,) and label.shape == ()
    assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long



def test_end_to_end_data_loading(mock_data_config):
    """测试 _load_data 是否能走通（含缓存逻辑）"""
    train_ds, test_ds, vocab = _load_data(mock_data_config)

    # 检查 dataset 长度
    assert len(train_ds) == 8
    assert len(test_ds) == 2
    assert isinstance(vocab, dict)
    assert "<pad>" in vocab and "<unk>" in vocab

    # 拿一个样本检查
    x, y = train_ds[0]
    assert x.shape == (mock_data_config.nlp_config.max_len,)
    assert y.shape == ()
    
    print(f"Sample input shape: {x.shape}, label: {y.item()}")


def test_build_datamodule(mock_exp_context):
    """端到端测试：build_datamodule → DataLoader → batch shape"""
    train_loader, val_loader, vocab_size = build_datamodule(mock_exp_context)

    # 检查 vocab_size 合理, 至少包含 <pad>, <unk> + 一些词
    assert vocab_size > 2 

    # 取一个 batch
    batch = next(iter(train_loader))
    inputs, labels = batch

    batch_size = mock_exp_context.train_config.batch_size
    max_len = mock_exp_context.data_config.nlp_config.max_len 

    print(f"Batch input shape: {inputs.shape}, labels shape: {labels.shape}")
    
    assert inputs.shape == (batch_size, max_len)
    assert labels.shape == (batch_size,)
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long



def test_cache_logic_works(mock_data_config):
    """验证缓存生成逻辑: 首次加载没有 cache → 下一次加载 cache"""
    cache_dir = Path(mock_data_config.data_dir) / "preprocessed"

    # 先确保无缓存
    for f in ["train.pkl", "test.pkl", "vocab.pkl"]:
        (cache_dir / f).unlink(missing_ok=True)

    # 第一次加载 ->  应生成 cache
    _load_data(mock_data_config)
    assert (cache_dir / "train.pkl").exists()
    assert (cache_dir / "vocab.pkl").exists()

    # 第二次加载->  应走缓存分支
    with patch("src.utils.logx.logger.debug") as mock_logger:  # 替换为实际 logger 路径
        _load_data(mock_data_config)
        mock_logger.assert_any_call("Loading text dataset from cache...")
        # 检查是否打印了缓存加载日志