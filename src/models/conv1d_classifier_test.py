from src.schemas.block_enums import PluginType
from src.models.conv1d_classifier import TextSeqClassifier

import torch
import pytest

# 通用测试参数
BATCH_SIZE = 4
SEQ_LEN = 20
VOCAB_SIZE = 1000
EMBEDDING_DIM = 64
HIDDEN_DIM = 32
OUTPUT_DIM = 10


class TestClassifiers:
    @pytest.fixture
    def input_ids(self):
        # 随机 token IDs，范围 [0, VOCAB_SIZE)
        return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    def test_text_seq_classifier_forward(self, input_ids):
        model = TextSeqClassifier(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            plugin_type=None,
        )
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), f"Expected {(BATCH_SIZE, OUTPUT_DIM)}, got {output.shape}"


    # 测试带插件
    @pytest.mark.parametrize("plugin_type", [None, PluginType.ID]) 
    def test_text_seq_classifier_with_plugin(self, input_ids, plugin_type):
        model = TextSeqClassifier(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            plugin_type=plugin_type,
        )
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    