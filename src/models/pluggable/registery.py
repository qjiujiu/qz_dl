from src.utils.logx import logger
from src.schemas.block_enums import PluginType
from typing import Callable, Dict
import torch.nn as nn



# 注册字典
ATTENTION_REGISTRY: Dict[PluginType, Callable] = {}


def register_pluggable_module(atten_type: PluginType):
    """装饰器，用于注册注意力模块"""
    def decorator(cls: Callable):
        ATTENTION_REGISTRY[atten_type] = cls
        return cls
    return decorator


def build_pluggable_block(block_key: PluginType, embed_dim: int):
    """根据 atten_key 获取对应的注意力模块"""
    if block_key not in ATTENTION_REGISTRY:
        logger.warning(f"Block type {block_key} not recognized, faulting to Identity.")
        return nn.Identity()
    
    return ATTENTION_REGISTRY[block_key](embed_dim)