from __future__ import annotations
from src.schemas.base_enums import OptimizerType, SchedulerType, LossType, TaskType
from src.schemas.block_enums import AttackType, AttenType
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional
from pathlib import Path
import uuid
import torch



# 基础组件配置
class NLPConfig(BaseModel):
    vocab_size: Optional[int]     = Field(278, description="词汇表大小")
    embedding_dim: Optional[int]  = Field(128, description="文本嵌入维度")
    max_len: Optional[int]        = Field(200, description="截断长度")
    atten: AttenType = AttenType.SELF


class VisionConfig(BaseModel):
    height: Optional[int]   = 224
    width: Optional[int]    = 224
    channels: Optional[int] = 3


class AdvConfig(BaseModel):
    enable: bool = Field(default=False, description="是否开启对抗训练")
    adv_type: AttackType = Field(default=AttackType.PGD, description="对抗训练方法类型")
    epsilon: float = Field(0.01, description="扰动半径约束")
    steps: int = Field(3, description="攻击迭代次数")
    alpha: float = Field(0.001, description="每次迭代的步长")



class ModelConfig(BaseModel):
    # 允许额外的字段，代替 explicit extra dict
    model_config = ConfigDict(extra='allow') 
    
    name: str = Field(..., description="模型架构名称")
    dropout_prob: float = Field(0.5, ge=0, le=1)
    
    # 统一使用 Path 类型，方便后续直接 .exists() 检查
    pretrained_dir: Optional[Path] = None 
    checkpoint_dir: Optional[Path] = Field(default=Path("./checkpoints"), description="模型保存目录")
    ouputs_log_dir: Optional[Path] = Field(default=Path("./outputs/log"), description="模型训练中间日志目录")
    ouputs_trace_dir: Optional[Path] = Field(default=Path("./outputs/trace"), description="模型训练记录目录")
    

class DataConfig(BaseModel):
    dataset_name: str
    data_dir: Path
    task_type: TaskType = Field(..., description="明确任务类型，用于校验")
    
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True

    # 使用 Optional 配合默认 None，这样比起 default_factory 更安全
    nlp_config: Optional[NLPConfig] = None
    vision_config: Optional[VisionConfig] = None

    @model_validator(mode='after')
    def check_config_consistency(self) -> DataConfig:
        """校验任务类型和配置是否匹配"""
        if self.task_type == TaskType.NLP and self.nlp_config is None:
            raise ValueError("Task is NLP but nlp_config is missing!")
        if self.task_type == TaskType.VISION and self.vision_config is None:
            raise ValueError("Task is Vision but vision_config is missing!")
        return self



class TrainConfig(BaseModel):
    batch_size: int = Field(128, ge=1)
    epochs: int = Field(50, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0, description="L2 正则化系数")
    momentum: float = Field(0.9, ge=0, le=1, description="动量因子，仅对 SGD 有效")
    
    optiz: OptimizerType = OptimizerType.ADAMW
    sched: Optional[str] = Field("cosine", description="学习率调度器")
    loss_fn: Optional[str] =   Field("cross_entropy", description="损失函数类型")
    
    
    seed: int = 3407
    device: str = Field("auto", description="cuda:0, cpu, mps, auto")


    @property
    def torch_device(self) -> torch.device:
        """辅助属性：直接获取 torch device 对象"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)



# 总入口
class ExpContext(BaseModel):
    model_config: ModelConfig
    data_config: DataConfig
    train_config: TrainConfig
    adv_config: AdvConfig = Field(default_factory=AdvConfig)

    # 实验元信息
    description: Optional[str] = Field(
        default="未填写任何描述", 
        description="实验描述/备注")
    
    task_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="实验唯一ID, 无需用户填写, 自动生成")
    
    # 允许通过 yaml 读取时忽略未知字段（为了兼容性）
    model_config = ConfigDict(extra='ignore')