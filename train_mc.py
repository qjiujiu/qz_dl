from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.datasets.instances import maldynamic
from src.light.multilabel_trainer import MultilabelTrainer
from src.models.build_model import build_model
import argparse
import importlib

import torch
# 释放缓存块,  重置峰值统计（不影响内存）

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats() 


# 动态导入 src.configs.{config_name}.ctx
def load_config_ctx(config_name: str, ctx_name: str) -> ExpContext:
    module_path = f"src.configs.{config_name}"
    try:
        config_module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Config module not found: '{module_path}'.\n"
            f"  -> Check if file `src/configs/{config_name}.py` exists.\n"
            f"  -> Error: {e}"
        ) from e

    # 检查模块是否有 `ctx` 属性
    if not hasattr(config_module, ctx_name):
        raise AttributeError(f"Module `{module_path}` does not define `{ctx_name}`!")
    
    ctx_obj = getattr(config_module, ctx_name)
    logger.success(f"Loaded config: `src.configs.{config_name}.{ctx_name}`")
    return ctx_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset with specified config and check batch shape")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Config module name under src.configs (e.g., 'lstm', 'cnn')")
    parser.add_argument("-n", "--name", type=str, default="ctx",
                        help="Variable name in the module (default: 'ctx')")
    
    args = parser.parse_args()
    ctx = load_config_ctx(args.config, args.name)
    
    model = build_model(ctx)
    train_loader, val_loader = maldynamic.build_datamodule(ctx)
    
    logger.info(model)
    logger.info(f"Training Start. context: {ctx}")
    
    tl = MultilabelTrainer(
        ctx=ctx, 
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
 
    
    tl.train()
    val_metrics = tl.evaluate()
    logger.info(f"Training Done. Val_metrics: {val_metrics}")