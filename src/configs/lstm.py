from src.utils.cuda import select_gpu
from src.schemas.context import ExpContext, NetworkConfig, DataConfig, NLPConfig, TrainConfig, AdvConfig
from src.schemas.base_enums import TaskType, OptimizerType
from src.schemas.block_enums import PluginType
from pathlib import Path

from src.models.lstm_classifer import LSTMSeqClassifier


ctx = ExpContext(
    description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + No Plugin", 
    network_config = NetworkConfig(
        name = LSTMSeqClassifier.__name__,
        dropout_prob = 0.3,
        num_classes=8,
        plugin_type = PluginType.ID,
    ),
    data_config = DataConfig(
        dataset_name = "malapi2019",
        data_dir = Path("./data/malapi2019"),
        task_type = TaskType.NLP,
        nlp_config = NLPConfig(
            vocab_size = 278, 
            embedding_dim = 256,
            max_len = 200,
        ),
    ),
    train_config = TrainConfig(
        batch_size = 512,
        epochs = 20,
        lr = 1e-3,
        optiz = OptimizerType.ADAM,
        device = select_gpu(),
    ),
    adv_config = AdvConfig(
      enable=False
    ),
)

ctx_sa = ctx.model_copy(deep=True)
ctx_sa.description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + 自注意力机制"
ctx_sa.network_config.plugin_type = PluginType.SA

ctx_sape = ctx.model_copy(deep=True)
ctx_sape.description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + 自注意力机制 + 位置编码"
ctx_sape.network_config.plugin_type = PluginType.SelfPE

ctx_pe = ctx.model_copy(deep=True)
ctx_pe.description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + 位置编码"
ctx_pe.network_config.plugin_type = PluginType.PosEnc

ctx_mlp = ctx.model_copy(deep=True)
ctx_mlp.description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + MLP注意力机制"
ctx_mlp.network_config.plugin_type = PluginType.MlpAtten

ctx_gas = ctx.model_copy(deep=True)
ctx_gas.description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + 普通高斯噪声"
ctx_gas.network_config.plugin_type = PluginType.GaussLinf


# 我们把不同预处理的逻辑视为完全不同的数据集 (数据版本化管理 Data-Versioning)
ctx_gc = ctx.model_copy(deep=True)
ctx_gc.description = f"MalAPI2019-GC 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + 自注意力机制"
ctx_gc.data_config.dataset_name = "malapi2019-gc"
ctx_gc.network_config.plugin_type = PluginType.ID