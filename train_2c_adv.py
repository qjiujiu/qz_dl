from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.datasets.instances import malanalysis
from src.light.adv_trainer import LatentPGDTrainer
from src.schemas.block_enums import AttackType
from src.models.build_model import build_model
import argparse
import importlib

import torch
# 释放缓存块,  重置峰值统计（不影响内存）

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats() 


# 动态导入 src.configs.{config_name}.ctx
def load_config_ctx(config_name: str, ctx_name: str, args: argparse.Namespace = None) -> ExpContext:
    """
    动态加载配置，并根据命令行参数(args)进行覆盖
    """
    module_path = f"src.configs.{config_name}"
    try:
        config_module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Config module not found: '{module_path}'.\n"
            f"  -> Check if file `src/configs/{config_name}.py` exists.\n"
            f"  -> Error: {e}"
        ) from e

    # 加载基础配置对象
    if not hasattr(config_module, ctx_name):
        raise AttributeError(f"Module `{module_path}` does not define `{ctx_name}`!")
    
    ctx_obj: ExpContext = getattr(config_module, ctx_name)
    logger.success(f"Loaded base config: `src.configs.{config_name}.{ctx_name}`")

    # 如果传入了 args，执行覆盖逻辑
    if args is not None:
        # 覆盖 Attack Type
        if  hasattr(args, 'adv_type') and args.adv_type:
            attack_type = AttackType[args.adv_type.upper()]
            ctx_obj.adv_config.adv_type = attack_type
            logger.info(f"CLI Override: adv_type -> {attack_type}")
            
        # 覆盖 Epsilon
        if hasattr(args, 'epsilon') and args.epsilon is not None:
            ctx_obj.adv_config.epsilon = args.epsilon
            logger.info(f"CLI Override: epsilon -> {ctx_obj.adv_config.epsilon}")

        # 覆盖 Steps
        if hasattr(args, 'steps') and args.steps is not None:
            ctx_obj.adv_config.steps = args.steps
            logger.info(f"CLI Override: steps -> {ctx_obj.adv_config.steps}")

        # 覆盖 Alpha
        if hasattr(args, 'alpha') and args.alpha is not None:
            ctx_obj.adv_config.alpha = args.alpha
            logger.info(f"CLI Override: alpha -> {ctx_obj.adv_config.alpha}")

        # 覆盖 epochs
        if hasattr(args, 'epochs') and args.epochs is not None:
            ctx_obj.train_config.epochs = args.epochs
            logger.info(f"CLI Override: epochs -> {ctx_obj.train_config.epochs}")
            
    return ctx_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adversarial Training")
    
    # 基础参数
    parser.add_argument("-c", "--config", type=str, required=True, help="Config module name")
    parser.add_argument("-n", "--name", type=str, default="ctx", help="Ctx variable name")
    
    # 覆盖参数 (Optional)
    parser.add_argument("--adv-type", type=str, choices=["FGM", "PGD"], help="Override attack type")
    parser.add_argument("--epsilon", type=float, help="Override perturbation epsilon")
    parser.add_argument("--steps", type=int, help="Override PGD iteration steps")
    parser.add_argument("--alpha", type=float, help="Override PGD step size")
    parser.add_argument("--epochs", type=int, help="Override Epochs")
    
    args = parser.parse_args()
    
    ctx = load_config_ctx(args.config, args.name, args)
    
    model = build_model(ctx)
    train_loader, val_loader = malanalysis.build_datamodule(ctx)
    
    logger.info(model)
    logger.info(f"Training Start. context: {ctx}")
    
    # 训练
    tl = LatentPGDTrainer(
        ctx=ctx, 
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
 
    
    tl.train()
    val_metrics = tl.evaluate()
    logger.info(f"Adv Training Done. Val_metrics: {val_metrics}")