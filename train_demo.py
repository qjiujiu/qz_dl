from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.datasets.instances import malapi2019
from src.light.trainer import Trainer
from src.models.build_model import build_model
import argparse
import importlib

# 动态导入 src.configs.{config_name}.ctx
def load_config_ctx(config_name: str) -> ExpContext:
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
    if not hasattr(config_module, "ctx"):
        raise AttributeError(f"Module `{module_path}` does not define `ctx`!")

    logger.success(f"Loaded config: `src.configs.{config_name}.ctx`")
    return config_module.ctx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset with specified config and check batch shape")
    parser.add_argument("-c", "--config",  type=str, help="Config module name under src.configs (e.g., 'lstm', 'cnn', 'transformer')")
    
    args = parser.parse_args()
    ctx = load_config_ctx(args.config)
    
    model = build_model(ctx)
    train_loader, val_loader, vocab = malapi2019.build_datamodule(ctx)
    
    logger.info(model)
    logger.info(f"Training Start. context: {ctx}")
    
    tl = Trainer(
        ctx=ctx, 
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
 
    
    tl.train()
    val_metrics = tl.evaluate()
    logger.info(f"Training Done. Val_metrics: {val_metrics}")