from src.utils.cuda import select_gpu
from src.schemas.context import ExpContext, NetworkConfig, DataConfig, NLPConfig, TrainConfig, AdvConfig
from src.schemas.base_enums import TaskType, OptimizerType
from src.schemas.block_enums import PluginType
from pathlib import Path



ctx = ExpContext(
    description = f"MalAPI2019 恶意软件API分类实验: Bert + No Plugin", 
    network_config = NetworkConfig(
        name = "Bert",
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
    # BERT 微调通常使用较小的学习率 (2e-5 ~ 5e-5), weight_decay 取用 1e-2
    train_config = TrainConfig(
        batch_size = 256,
        epochs = 20,
        lr = 2e-5,
        weight_decay=1e-2, 
        optiz = OptimizerType.ADAMW,
        device = select_gpu(),
    ),
    adv_config = AdvConfig(
      enable=False
    ),
)