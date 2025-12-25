from src.models.tcn_classifer import TemporalBlock, TCN, TCNSeqClassifier
import pytest
import torch
import torch.nn as nn



class TestTemporalBlock:
    @pytest.mark.parametrize("kernel_size", [3, 5])
    def test_output_shape_consistency(self, kernel_size):
        """测试 TemporalBlock 是否保持输入输出长度一致 (Same Padding)"""
        batch, channels, seq_len = 4, 32, 50
        dilation = 1
        # 根据 TCN 逻辑计算 padding
        padding = (kernel_size - 1) * dilation // 2
        
        block = TemporalBlock(
            n_inputs=channels,
            n_outputs=channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding
        )
        
        x = torch.randn(batch, channels, seq_len)
        out = block(x)
        
        # TCN 核心要求：卷积后长度不变
        assert out.shape == (batch, channels, seq_len)


    def test_residual_projection(self):
        """测试当输入输出通道不一致时，1x1卷积是否生效"""
        batch, in_c, out_c, seq_len = 2, 16, 32, 20
        block = TemporalBlock(in_c, out_c, kernel_size=3, stride=1, dilation=1, padding=1)
        
        x = torch.randn(batch, in_c, seq_len)
        out = block(x)
        
        # 应该初始化 downsample layer
        assert out.shape == (batch, out_c, seq_len)
        assert block.downsample is not None 



class TestTCN:
    def test_even_kernel_size_error(self):
        """测试偶数卷积核是否会抛出 ValueError"""
        with pytest.raises(ValueError, match="must be odd"):
            TCN(num_inputs=32, num_channels=[32], kernel_size=2)

    def test_receptive_field_structure(self):
        """测试多层 TCN 的构建结构"""
        num_inputs = 64
        channels = [64, 128, 256] # 3层
        model = TCN(num_inputs, channels, kernel_size=3)
        
        # 检查是否生成了正确数量的 TemporalBlock
        assert len(model.network) == 3
        
        # 检查输入输出流
        batch, seq_len = 4, 100
        x = torch.randn(batch, num_inputs, seq_len)
        out = model(x)
        
        # 输出通道应该是 channels 列表的最后一个元素
        assert out.shape == (batch, channels[-1], seq_len)
        


class TestTCNSeqClassifier:
    @pytest.fixture
    def model_config(self):
        return {
            "vocab_size": 1000,
            "embedding_dim": 128,
            "tcn_channels": [128, 256], 
            "output_dim": 2,
            "dropout": 0.1
        }

    @pytest.fixture
    def model(self, model_config):
        return TCNSeqClassifier(**model_config)

    def test_forward_pass_shape(self, model, model_config):
        """测试完整的前向传播输出维度"""
        batch_size = 8
        seq_len = 50
        
        # 构造模拟的 token 输入 [Batch, Seq_Len]
        x = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len))
        
        output = model(x)
        
        # 期望输出: [Batch, Output_Dim]
        expected_shape = (batch_size, model_config["output_dim"])
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()

    def test_backward_propagation(self, model, model_config):
        """测试梯度反向传播 (确保没有 detach 或者 inplace error)"""
        batch_size = 4
        seq_len = 30
        
        x = torch.randint(0, model_config["vocab_size"], (batch_size, seq_len))
        target = torch.randint(0, model_config["output_dim"], (batch_size,))
        
        # 前向
        logits = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)
        
        # 反向
        loss.backward()
        
        # 检查 Embedding 层是否有梯度
        assert model.embedding.weight.grad is not None
        # 检查 FC 层是否有梯度
        assert model.fc.weight.grad is not None
        # 检查梯度数值总和不等于 0 
        assert torch.sum(torch.abs(model.fc.weight.grad)) > 0

  
    def test_max_pooling_logic(self, model_config):
        """
        验证 Max Pooling 逻辑是否正确处理了维度
        这里我们手动 hack 一下 embedding 层来验证 pooling 行为
        """
        model = TCNSeqClassifier(**model_config)
        
        # 模拟一个 batch=1, seq_len=3 的输入
        # 我们希望验证无论序列多长，最后输出都是 [Batch, Out_Dim]
        
        # 长序列
        x = torch.randint(0, 10, (1, 300)) 
        print(x.shape)
        y = model(x)
        print(y.shape)
        assert y.shape == (1, model_config["output_dim"])
        
        # 短序列
        x_short = torch.randint(0, 10, (1, 10)) 
        print(x.shape)
        y_short = model(x_short)
        print(y.shape)
        assert y_short.shape == (1, model_config["output_dim"])