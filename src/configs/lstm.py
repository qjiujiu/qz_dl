from src.utils.cuda import select_gpu
from src.configs.utils import with_varients
from src.schemas.base_enums import TaskType, OptimizerType
from src.schemas.context import (
    ExpContext, 
    NetworkConfig, TrainConfig, 
    DataConfig, NLPConfig, 
    AdvConfig,
)
from src.schemas.block_enums import PluginType
from src.models.lstm_classifer import LSTMSeqClassifier
from pathlib import Path


nc8 = NetworkConfig(
    name = LSTMSeqClassifier.__name__,
    dropout_prob = 0.3,
    num_classes=8,
    plugin_type = PluginType.ID,
)

tc = TrainConfig(
    batch_size = 512,
    epochs = 20,
    lr = 1e-3,
    optiz = OptimizerType.ADAM,
    device = select_gpu(),
)


dc8 =  DataConfig(
    dataset_name = "malapi2019",
    data_dir = Path("./data/malapi2019"),
    task_type = TaskType.NLP,
    nlp_config = NLPConfig(
        vocab_size = 278, 
        embedding_dim = 256,
        max_len = 200,
    ),
)


ctx8 = ExpContext(
    description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + No Plugin", 
    network_config = nc8,
    data_config = dc8, 
    train_config = tc,
    adv_config = AdvConfig(
      enable=False
    ),
)



ctx8_sa, ctx8_sape, ctx8_pe, ctx8_mlp, ctx8_gas = (    
    with_varients(ctx8, model_name = LSTMSeqClassifier.__name__, 
        suffix="自注意力机制", 
        plugin=PluginType.SA), 
    with_varients(ctx8, model_name = LSTMSeqClassifier.__name__, 
        suffix="自注意力机制 + 位置编码", 
        plugin=PluginType.SelfPE),
    with_varients(ctx8, model_name = LSTMSeqClassifier.__name__, 
        suffix="位置编码", 
        plugin=PluginType.PosEnc), 
    with_varients(ctx8, model_name = LSTMSeqClassifier.__name__, 
        suffix="MLP注意力机制", 
        plugin=PluginType.MlpAtten), 
    with_varients(ctx8, model_name = LSTMSeqClassifier.__name__, 
        suffix="普通高斯噪声", 
        plugin=PluginType.GaussLinf),
)


# 我们把不同预处理的逻辑视为完全不同的数据集 (数据版本化管理 Data-Versioning)
ctx_gc = with_varients(ctx8, model_name = LSTMSeqClassifier.__name__,  
        suffix="自注意力机制", 
        plugin=PluginType.ID, 
        dataset_name="malapi2019-gc")



# 其它数据集(二分类)
nc2 = NetworkConfig(
    name = LSTMSeqClassifier.__name__,
    dropout_prob = 0.3,
    num_classes=2,
    plugin_type = PluginType.ID,
)

dc2 =  DataConfig(
    dataset_name = "malanalysis",
    data_dir = Path("./data/malware-analysis-datasets-api-call-sequences"),
    task_type = TaskType.NLP,
    nlp_config = NLPConfig(
        vocab_size = 307, 
        embedding_dim = 256,
        max_len = 100,
    ),
)

ctx2 = ExpContext(
    description = f"MalAPI2019 恶意软件API分类实验: {LSTMSeqClassifier.__name__} + No Plugin", 
    network_config = nc2,
    data_config = dc2, 
    train_config = tc,
    adv_config = AdvConfig(
      enable=False
    ),
)


# 其它数据集(多标签多分类)

