from src.utils.logx import logger
from src.datasets.instances import malapibert 
from src.schemas.context import ExpContext 
from src.light.bert_trainer import BertTrainer 
from src.configs.bert import ctx

from transformers import AutoModelForSequenceClassification

import torch
import torch.nn as nn

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats() 


# NOTE 此处区分了全量微调与局部微调两种方式
def build_bert_model(num_labels: int, freeze_backbone: bool = False) -> nn.Module:
    """ 构建 BERT 类模型
    """
    
    logger.info(f"Building Bert Model, Freeze Backbone: {freeze_backbone}")
    
    # 加载预训练模型
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = 'microsoft/codebert-base',      # 获取预训练模型名称
        num_labels = num_labels,                                        # 分类头的分类个数
        weights_only = False                                            # 解决 Pickle 安全警告
    )

    # 如果冻结基座模型的参数, 冻结所有参数
    if not freeze_backbone:
        logger.info("Full Fine-tuning enabled. All parameters are trainable.")
        return model

    # NOTE 解冻分类头: HuggingFace 分类头通常命名 'classifier' (RoBERTa/BERT) 或 'score'
    # NOTE 遍历 named_parameters，只要名字里包含分类头关键词则将其设为 True
    for param in model.parameters():
        param.requires_grad = False
        
    trainable_params = 0
    for name, param in model.named_parameters():
        if any(k in name for k in ["classifier", "score", "cls"]):
            param.requires_grad = True
            trainable_params += param.numel()
    
    logger.info(f"Backbone frozen. Trainable parameters (Head only): {trainable_params}")
    return model



def run_training(ctx: ExpContext):
    logger.info(f"Loading Pretrained Model: {ctx.network_config.name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path = 'microsoft/codebert-base',
        num_labels=ctx.network_config.num_classes,
        weights_only=False, 
    )
        
    # 加载数据
    logger.info("Loading Datasets...")
    train_loader, val_loader = malapibert.build_datamodule(ctx)
    
    # 实例化 BertTrainer
    trainer = BertTrainer(ctx, model, train_loader, val_loader)
    
    # 开始训练
    trainer.train()



if __name__ == "__main__":
    run_training(ctx)