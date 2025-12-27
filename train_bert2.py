from src.utils.logx import logger
from src.datasets.instances import malapibert 
from src.schemas.context import ExpContext 
from src.light.bert_trainer import BertTrainer 
from src.configs.bert import ctx

from transformers import AutoModelForSequenceClassification

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