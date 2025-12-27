from src.schemas.context import ExpContext
from src.schemas.block_enums import PluginType
from typing import Optional



def with_varients(base: ExpContext, model_name: str, suffix: str, plugin: Optional[PluginType] = None, dataset_name: Optional[str] = None):
    nc = base.network_config.model_copy(deep=True)
    dc = base.data_config.model_copy(deep=True)
    
    if plugin:
        nc.plugin_type = plugin
    
    if dataset_name:
        dc.dataset_name = dataset_name

    return base.model_copy(update={
        "description": f"Dataset: {dc.dataset_name} 恶意软件API分类实验: {model_name} + {suffix}",
        "network_config": nc,
        "data_config": dc,
    }, deep=True)
    