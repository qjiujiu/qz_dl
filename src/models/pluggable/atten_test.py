import torch
import pytest

from src.schemas.block_enums import PluginType
from src.models.pluggable.registery import build_pluggable_block
from src.models.pluggable.atten import (
    MLPAtten,
    SinusoidalPE,
    SelfAtten,
    SelfAttenWithSinusoidalPE,
)


@pytest.fixture
def random_noise_data():
    """生成一些随机噪声数据作为测试输入"""
    batch_size, seq_len, embed_dim = 8, 10, 64
    X = torch.randn(batch_size, seq_len, embed_dim)  
    return X


class TestAttentionModules:
    def test_mlp_attention(self, random_noise_data):
        """测试 MLPAttention 模块"""
        attention: MLPAtten = build_pluggable_block(PluginType.MlpAtten, embed_dim=64)
        output = attention(random_noise_data)
        
        # 输出形状应与输入形状一致：[batch_size, seq_len, embed_dim]
        print(output.shape)
        assert output.shape == random_noise_data.shape, f"Expected shape {random_noise_data.shape}, but got {output.shape}"
    
    def test_sinusoidal_pe(self, random_noise_data):
        """测试 SinusoidalPE 模块"""
        attention: SinusoidalPE = build_pluggable_block(PluginType.PosEnc, embed_dim=64)
        output = attention(random_noise_data)
        
        # 输出形状应与输入形状一致：[batch_size, seq_len, embed_dim]
        print(output.shape)
        assert output.shape == random_noise_data.shape, f"Expected shape {random_noise_data.shape}, but got {output.shape}"
    
    def test_self_attention(self, random_noise_data):
        """测试 SelfAttention 模块"""
        attention: SelfAtten = build_pluggable_block(PluginType.SA, embed_dim=64)
        output = attention(random_noise_data)
        
        # 输出形状应与输入形状一致：[batch_size, seq_len, embed_dim]
        print(output.shape)
        assert output.shape == random_noise_data.shape, f"Expected shape {random_noise_data.shape}, but got {output.shape}"
    
    def test_self_attention_with_pe(self, random_noise_data):
        """测试 SelfAttentionWithSinusoidalPE 模块"""
        attention: SelfAttenWithSinusoidalPE = build_pluggable_block(PluginType.SelfPE, embed_dim=64)
        output = attention(random_noise_data)
        
        # 输出形状应与输入形状一致：[batch_size, seq_len, embed_dim]
        print(output.shape)
        assert output.shape == random_noise_data.shape, f"Expected shape {random_noise_data.shape}, but got {output.shape}"