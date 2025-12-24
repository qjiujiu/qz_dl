from src.utils.logx import logger
from src.trainer.trainer import Trainer
from src.schemas.context import ExpContext, NetworkConfig, DataConfig, TrainConfig, TaskType, NLPConfig
from src.schemas.base_enums import OptimizerType, LossType, SchedulerType

from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from torch import nn
import torch
import pytest
import numpy as np



# 创建简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# 测试数据和数据加载器
@pytest.fixture
def dummy_data():
    # 生成一些随机数据
    X_train = torch.randn(100, 10)         # 训练样本，每个样本10个特征
    y_train = torch.randint(0, 2, (100,))  # 随机的标签，0或1
    X_val = torch.randn(20, 10)            # 验证样本
    y_val = torch.randint(0, 2, (20,))     # 随机的验证标签
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    return train_loader, val_loader

# 实例化ExpContext和Trainer的测试数据
@pytest.fixture
def exp_context():
    # 创建一个简单的模型配置和训练配置
    model_config = NetworkConfig(name="simple_model", dropout_prob=0.5)
    data_config = DataConfig(
        dataset_name="dummy_dataset", 
        data_dir="data", 
        task_type=TaskType.NLP, 
        nlp_config=NLPConfig()
    )
    train_config = TrainConfig(
        batch_size=32, epochs=1, lr=1e-3,
        weight_decay=1e-4,
        seed=3407,
        device="cpu", 
        optiz=OptimizerType.ADAM, 
        sched=SchedulerType.LINEAR,
        loss_fn=LossType.CROSS_ENTROPY, 
    )
    
    # 创建ExpContext
    return ExpContext(
        network_config=model_config,
        data_config=data_config,
        train_config=train_config,
    )


@pytest.fixture
def trainer(exp_context: ExpContext, dummy_data) -> Trainer:
    model = SimpleModel()
    train_loader, val_loader = dummy_data
    return Trainer(context=exp_context, model=model, train_loader=train_loader, val_loader=val_loader)




class TestTrainer:
    def test_train_method(self, trainer: Trainer):
        """测试train方法，确保训练可以执行"""
        trainer.train()  # 训练一次
        assert len(trainer.history['train_loss']) == 1   # 至少执行了1个epoch
        assert len(trainer.history['val_metrics']) == 1  # 至少执行了1次验证

    def test_evaluate_method(self, trainer: Trainer):
        """测试evaluate方法，确保评估能够计算并返回指标"""
        val_metrics = trainer.evaluate()
        assert "acc" in val_metrics       # 结果应包含准确率
        assert val_metrics["acc"] >= 0.0  # 精度应为正数

    def test_save_checkpoint(self, trainer: Trainer):
        """测试保存检查点"""
        trainer._save_checkpoint(suffix="epoch-1")
        
        # 检查点文件应已保存
        checkpoint_path = trainer.ctx.network_config.checkpoint_dir / f"simple_model-epoch-1.pth"
        assert checkpoint_path.exists()  

    def test_save_trace(self, trainer: Trainer, tmp_path: Path):
        """测试保存训练日志"""
        print(tmp_path)
        
        # 模拟输出目录
        trainer.ctx.network_config.ouputs_trace_dir = tmp_path
        trainer._save_trace()
        
        # 日志文件应已保存
        trace_path = tmp_path / f"{trainer.task_id}.json"
        assert trace_path.exists()  