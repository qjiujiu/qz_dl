from src.schemas.context import ExpContext, NetworkConfig, DataConfig, NLPConfig, TrainConfig, AdvConfig
from src.schemas.base_enums import TaskType, OptimizerType
from src.schemas.block_enums import PluginType
from pathlib import Path

from src.models.lstm_classifer import LSTMSeqClassifier

ctx = ExpContext(
    description = "MalAPI2019 恶意软件API分类实验: LSTMSeqClassifier + No Plugin", 
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
        batch_size = 8,
        epochs = 5,
        lr = 1e-3,
        optiz = OptimizerType.ADAM,
        device = "cuda",
    ),
    adv_config = AdvConfig(
      enable=False
    ),
)
