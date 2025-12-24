from src.schemas.context import TrainConfig
from src.schemas.base_enums import OptimizerType, SchedulerType, LossType
from src.trainer.build_helper import build_loss_fn, build_optimizer, build_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pytest
import torch.nn as nn
import torch.optim as optim


@pytest.fixture
def dummy_model():
    """最简单的模型用于测试 optimizer 参数"""
    return nn.Linear(10, 2)


@pytest.fixture
def base_cfg() -> TrainConfig:
    """一个最小可用的 TrainConfig（按你真实字段补齐）"""
    return TrainConfig(
        batch_size=32,
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
        momentum=0.9,
        optiz=OptimizerType.ADAM,
        sched=SchedulerType.LINEAR,
        loss_fn=LossType.CROSS_ENTROPY,
    )



class TestBuildLossFn:
    def test_build_loss_cross_entropy(self, base_cfg):
        """正确返回 CrossEntropyLoss"""
        loss_fn = build_loss_fn(base_cfg)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

    def test_build_loss_unknown_fallback(self, base_cfg, caplog):
        """loss_fn 未知时 fallback 走到 CrossEntropy 并且给出 1 warning"""
        caplog.clear()

        # 模拟非法 loss_fn
        base_cfg.loss_fn = "UNKNOWN_LOSS"

        loss_fn = build_loss_fn(base_cfg)
        assert isinstance(loss_fn, nn.CrossEntropyLoss)

        # 检查 warning 日志
        assert any("falling back to CrossEntropy" in record.message for record in caplog.records)



class TestBuildOptimizer:
    def test_build_optimizer_adam(self, base_cfg, dummy_model):
        base_cfg.optiz = OptimizerType.ADAM
        opt = build_optimizer(base_cfg, dummy_model)
        assert isinstance(opt, optim.Adam)

    def test_build_optimizer_adamw(self, base_cfg, dummy_model):
        base_cfg.optiz = OptimizerType.ADAMW
        opt = build_optimizer(base_cfg, dummy_model)
        assert isinstance(opt, optim.AdamW)

    def test_build_optimizer_sgd(self, base_cfg, dummy_model):
        base_cfg.optiz = OptimizerType.SGD
        opt = build_optimizer(base_cfg, dummy_model)
        assert isinstance(opt, optim.SGD)
        # momentum 也应该生效
        assert opt.param_groups[0]["momentum"] == pytest.approx(base_cfg.momentum)

    def test_build_optimizer_unknown_fallback(self, base_cfg, dummy_model, caplog):
        """未知 optimizer 类型应 fallback 到 Adam 并 warning"""
        caplog.clear()

        base_cfg.optiz = "UNKNOWN_OPT"
        opt = build_optimizer(base_cfg, dummy_model)

        assert isinstance(opt, optim.Adam)
        assert any("defaulting to Adam" in record.message for record in caplog.records)



class TestBuildScheduler:
    def test_build_scheduler_none_returns_linear(self, base_cfg: TrainConfig, dummy_model):
        opt = build_optimizer(base_cfg, dummy_model)

        base_cfg.sched = None
        sched = build_scheduler(base_cfg, opt)
        assert isinstance(sched, LambdaLR)

        base_cfg.sched = SchedulerType.LINEAR
        assert isinstance(sched, LambdaLR)

    def test_build_scheduler_cosine(self, base_cfg: TrainConfig, dummy_model):
        """COSINE 返回 CosineAnnealingLR"""
        opt = build_optimizer(base_cfg, dummy_model)

        base_cfg.sched = SchedulerType.COSINE
        sch = build_scheduler(base_cfg, opt)

        assert isinstance(sch, CosineAnnealingLR)
        assert sch.T_max == base_cfg.epochs


    def test_build_scheduler_linear(self, base_cfg: TrainConfig, dummy_model):
        """LINEAR 返回 LambdaLR"""
        opt = build_optimizer(base_cfg, dummy_model)

        base_cfg.sched = SchedulerType.LINEAR
        sch = build_scheduler(base_cfg, opt)

        assert isinstance(sch, LambdaLR)

        lr0 = sch.get_last_lr()[0]
        sch.step()
        lr1 = sch.get_last_lr()[0]
        
        print(f"lr0: {lr0})")
        print(f"lr1: {lr1})")
        
        assert lr0 != lr1
        

    def test_build_scheduler_unknown_fallback(self, base_cfg: TrainConfig, dummy_model, caplog):
        """未知 scheduler 类型 fallback 到 LINEAR 并 warning"""
        caplog.clear()

        opt = build_optimizer(base_cfg, dummy_model)

        base_cfg.sched = "UNKNOWN_SCHED"
        sch = build_scheduler(base_cfg, opt)

        assert isinstance(sch, LambdaLR)
        assert any("defaulting to Linear" in record.message for record in caplog.records)