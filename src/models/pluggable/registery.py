from src.utils.logx import logger
from src.schemas.block_enums import PluginType
from typing import Callable, Dict
import torch.nn as nn



# 注册字典
ATTENTION_REGISTRY: Dict[PluginType, Callable] = {}


def register_pluggable_module(atten_type: PluginType):
    """装饰器，用于注册注意力模块"""
    def decorator(block_cls: Callable):
        ATTENTION_REGISTRY[atten_type] = block_cls
        return block_cls
    return decorator


def build_pluggable_block(block_key: PluginType, embed_dim: int):
    """ 根据 atten_key 获取对应的注意力模块, 只有导入的时候会注册装饰器进行注册, 
        因此 from ... import ... 不可删除, 其作用是确保相关模块已被导入, 若删除, 会进入恒等映射兜底!
    """
    from src.models.pluggable.atten import (
        MLPAtten,
        SinusoidalPE, 
        SelfAtten, 
        SelfAttenWithSinusoidalPE
    )
    
    if ATTENTION_REGISTRY.get(block_key) is None:
        logger.warning(f"Block type {block_key} not recognized, faulting to Identity.")
        return nn.Identity()
    
    return ATTENTION_REGISTRY[block_key](embed_dim)