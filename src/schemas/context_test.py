
from src.schemas.base_enums import TaskType, OptimizerType, SchedulerType, LossType
from src.schemas.block_enums import AttenType, AttackType
from src.schemas.context import (
    ExpContext, NetworkConfig, DataConfig, TrainConfig,
    NLPConfig, VisionConfig, AdvConfig,
)

from pathlib import Path
from pydantic import ValidationError
import pytest
import torch



@pytest.fixture
def base_model_config():
    return NetworkConfig(
        name="lenet",
        dropout_prob=0.3,
    )


@pytest.fixture
def base_train_config():
    return TrainConfig(
        batch_size=32,
        epochs=3,
        lr=1e-3,
        device="cpu",
    )


@pytest.fixture
def base_nlp_data_config():
    return DataConfig(
        dataset_name="dummy_nlp_dataset",
        data_dir=Path("./data"),
        task_type=TaskType.NLP,
        nlp_config=NLPConfig(),
    )


@pytest.fixture
def base_vision_data_config():
    return DataConfig(
        dataset_name="dummy_vision_dataset",
        data_dir=Path("./data"),
        task_type=TaskType.VISION,
        vision_config=VisionConfig(),
    )
    
class TestDataConfigValidator:
    def test_nlp_task_missing_nlp_config_should_raise(self):
        """NLP 任务缺少 nlp_config 必须报错"""
        with pytest.raises(ValidationError):
            DataConfig(
                dataset_name="dummy_nlp_dataset",
                data_dir=Path("./data"),
                task_type=TaskType.NLP,
                nlp_config=None,  # 缺失
            )

    def test_vision_task_missing_vision_config_should_raise(self):
        """VISION 任务缺少 vision_config 必须报错"""
        with pytest.raises(ValidationError):
            DataConfig(
                dataset_name="dummy_vision_dataset",
                data_dir=Path("./data"),
                task_type=TaskType.VISION,
                vision_config=None,  # 缺失
            )

    def test_nlp_task_with_nlp_config_ok(self, base_nlp_data_config):
        """NLP 任务提供 nlp_config 正常"""
        assert base_nlp_data_config.nlp_config is not None
        assert base_nlp_data_config.task_type == TaskType.NLP

    def test_vision_task_with_vision_config_ok(self, base_vision_data_config):
        """VISION 任务提供 vision_config 正常"""
        assert base_vision_data_config.vision_config is not None
        assert base_vision_data_config.task_type == TaskType.VISION
        



class TestTrainConfigTorchDevice:
    def test_device_auto_returns_cpu_or_cuda(self):
        """device=auto 时，返回 cpu 或 cuda"""
        tc = TrainConfig(device="auto")
        dev = tc.torch_device

        assert isinstance(dev, torch.device)
        assert dev.type in {"cpu", "cuda"}

        # 进一步验证逻辑一致性
        if torch.cuda.is_available():
            assert dev.type == "cuda"
        else:
            assert dev.type == "cpu"

    def test_device_cpu(self):
        tc = TrainConfig(device="cpu")
        assert tc.torch_device.type == "cpu"

    def test_device_cuda_string(self):
        """即便环境没有 GPU，torch.device('cuda:0') 也能构造出来（但不能真正用）"""
        tc = TrainConfig(device="cuda:0")
        assert tc.torch_device.type == "cuda"
        assert tc.torch_device.index == 0



class TestModelConfig:
    def test_model_name_required(self):
        """name 必填，缺失应报错"""
        with pytest.raises(ValidationError):
            NetworkConfig(dropout_prob=0.3)

    def test_checkpoint_dir_default(self):
        mc = NetworkConfig(name="lenet", dropout_prob=0.3)
        assert mc.checkpoint_dir == Path("./checkpoints")



class TestExpContext:
    def test_exp_context_defaults(self, base_model_config, base_nlp_data_config, base_train_config):
        """ExpContext 默认字段正确"""
        ctx = ExpContext(
            network_config=base_model_config,
            data_config=base_nlp_data_config,
            train_config=base_train_config,
        )

        # task_id 自动生成
        assert isinstance(ctx.task_id, str)
        assert len(ctx.task_id) > 0

        # description 默认值
        assert ctx.description == "未填写任何描述"

        # adv_config 默认生成
        assert isinstance(ctx.adv_config, AdvConfig)
        assert ctx.adv_config.enable is False
        assert ctx.adv_config.adv_type == AttackType.PGD

    def test_exp_context_with_custom_description(self, base_model_config, base_nlp_data_config, base_train_config):
        """自定义 description"""
        ctx = ExpContext(
            network_config=base_model_config,
            data_config=base_nlp_data_config,
            train_config=base_train_config,
            description="这是一个测试实验",
        )
        assert ctx.description == "这是一个测试实验"

    def test_exp_context_with_vision_task(self, base_model_config, base_vision_data_config, base_train_config):
        """VISION 任务的 ExpContext 正常构造"""
        ctx = ExpContext(
            network_config=base_model_config,
            data_config=base_vision_data_config,
            train_config=base_train_config,
        )
        assert ctx.data_config.task_type == TaskType.VISION
        assert ctx.data_config.vision_config is not None
        