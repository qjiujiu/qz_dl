from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block
from src.models.pluggable.atten import MLPAtten, SinusoidalPE, SelfAtten, SelfAttenWithSinusoidalPE
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def random_noise_data():
    """生成一些随机噪声数据作为测试输入"""
    batch_size, seq_len, embed_dim = 8, 10, 64
    X = torch.randn(batch_size, seq_len, embed_dim)  
    return X


class TestBuildPluggableBlock:
    def test_build_pluggable_block_valid(self):
        """测试构建无效的即插即用模块时抛出异常"""
        block = build_pluggable_block(None, embed_dim=64)
        print(block)
        
        assert isinstance(block, nn.Module)
        assert isinstance(block, nn.Identity)
        
        for p in PluginType.__members__.values():
            if p == PluginType.ID:
                continue
            block = build_pluggable_block(p, embed_dim=64) 
            print(block)
            assert isinstance(block, nn.Module) 
            assert not isinstance(block, nn.Identity)
        
    
    @pytest.mark.parametrize("plugin_type, expected_class", [
        (PluginType.MlpAtten, MLPAtten),
        (PluginType.PosEnc, SinusoidalPE),
        (PluginType.SA, SelfAtten), 
        (PluginType.SelfPE, SelfAttenWithSinusoidalPE),
    ])
    def test_build_pluggable_block_invalid(self, plugin_type, expected_class):
        """测试构建有效的即插即用模块"""
        block = build_pluggable_block(plugin_type, embed_dim=64)
        assert isinstance(block, nn.Module)
        assert isinstance(block, expected_class)
        
        