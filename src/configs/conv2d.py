from src.schemas.context import ExpContext, NetworkConfig, DataConfig, NLPConfig, TrainConfig, AdvConfig
from src.schemas.base_enums import TaskType, OptimizerType
from src.schemas.block_enums import PluginType
from pathlib import Path

from src.models.conv2d_classifier import Seq2ImageClassifier

ctx = ExpContext(
    description = "MalAPI2019 恶意软件API分类实验: Seq2ImageClassifier + No Plugin", 
    network_config = NetworkConfig(
        name = Seq2ImageClassifier.__name__,
        dropout_prob = 0.3,
        num_classes = 8,
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
        batch_size = 64,
        epochs = 30,
        lr = 1e-3,
        optiz = OptimizerType.ADAM,
        device = "cuda:1",
    ),
    adv_config = AdvConfig(
      enable=False
    ),
)


ctx_sa = ctx.model_copy(deep=True)
ctx_sa.description = "MalAPI2019 恶意软件API分类实验: Seq2ImageClassifier + 自注意力机制"
ctx_sa.network_config.plugin_type = PluginType.SA

ctx_sape = ctx.model_copy(deep=True)
ctx_sape.description = "MalAPI2019 恶意软件API分类实验: Seq2ImageClassifier + 自注意力机制 + 位置编码"
ctx_sape.network_config.plugin_type = PluginType.SelfPE

ctx_pe = ctx.model_copy(deep=True)
ctx_pe.description = "MalAPI2019 恶意软件API分类实验: Seq2ImageClassifier + 位置编码"
ctx_pe.network_config.plugin_type = PluginType.PosEnc

ctx_mlp = ctx.model_copy(deep=True)
ctx_mlp.description = "MalAPI2019 恶意软件API分类实验: Seq2ImageClassifier + MLP注意力机制"
ctx_mlp.network_config.plugin_type = PluginType.MlpAtten