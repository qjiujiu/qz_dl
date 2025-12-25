from src.schemas.block_enums import PluginType
from src.models.lstm_classifer import LSTMSeqClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch
import pytest


@pytest.fixture
def num_classes() -> int:
    """返回分类数量"""
    return 2 

@pytest.fixture
def dummy_data(num_classes: int):
    """生成一些随机数据作为训练数据"""
    X_train = torch.randint(0, 100, (32, 10)) 
    y_train = torch.randint(0, 32, (32,)) 
    
    print(X_train.shape)
    print(y_train.shape)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    return train_loader


@pytest.fixture
def model():
    """实例化 LSTMSeqClassifier 模型"""
    return LSTMSeqClassifier(
        vocab_size=100,      # 假设词汇表大小为 100
        embedding_dim=64,    # 词向量维度
        hidden_dim=128,      # LSTM 隐藏层维度
        output_dim=2,        # 二分类任务
        bidirectional=True,  # 使用双向 LSTM
        layers=2,            # LSTM 层数为 2
        dropout=0.3          # dropout 概率
    )


class TestLSTMSeqClassifier:
    def test_plugin_integration(self):
        """测试即插即用模块的集成是否正确"""
        model = LSTMSeqClassifier(
            vocab_size=100,
            embedding_dim=64,
            hidden_dim=128,
            output_dim=2,
            bidirectional=True,
            layers=2,
            dropout=0.3,
            plugin_type=None  # 不使用插件
        )
        
        assert isinstance(model.plugin, torch.nn.Identity)
        
        
        for plugin in [PluginType.MlpAtten, PluginType.PosEnc, PluginType.SA, PluginType.SelfPE]:
            model = LSTMSeqClassifier(
                vocab_size=100,
                embedding_dim=64,
                hidden_dim=128,
                output_dim=2,
                bidirectional=True,
                layers=2,
                dropout=0.3,
                plugin_type=plugin
            )
            assert not isinstance(model.plugin, torch.nn.Identity) 
        
    def test_forward_pass(self, model, dummy_data, num_classes):
        """测试前向传播，确保输出形状正确"""
        # 获取批次数据
        data_iter = iter(dummy_data)
        X_batch, y_batch = next(data_iter)  #
        
        # 将数据传入模型
        model.eval()             # 切换到评估模式
        output = model(X_batch)  # 前向传播
        
        # 输出形状应为 [batch_size, output_dim]，即 [8, num_classes]
        # 断言输出的形状符合预期
        assert output.shape == (X_batch.size(0), num_classes) 
    
    def test_trainable_parameters(self, model):
        """检查模型是否有可训练参数"""
        # 获取模型中的参数
        params = list(model.parameters())
        
        # 断言模型中至少有一个可训练的参数
        assert len(params) > 0
        for param in params:
            assert param.requires_grad 
    
    
    # 测试不同长度的序列
    @pytest.mark.parametrize("seq_len", [5, 10, 15])  
    def test_forward_pass_with_random_input(self, model, seq_len, num_classes):
        """测试模型的前向传播，确保输出形状正确"""
        
        model.eval()  
        batch_size = 8
        X_batch = torch.randint(0, 100, (batch_size, seq_len))  # 随机生成输入数据
        
        output = model(X_batch)  
        # 输出形状应为 [batch_size, output_dim]，i.e [8, num_classes]
        assert output.shape == (batch_size, num_classes)
    
    
    def test_checkpoint_saving(self, model, dummy_data, tmp_path):
        """测试模型检查点是否能正确保存"""
        model.eval()  # 评估模式
        data_iter = iter(dummy_data)
        X_batch, y_batch = next(data_iter)
        
        # 保存检查点
        checkpoint_path = tmp_path / "model_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # 断言检查点文件是否存在
        assert checkpoint_path.exists()