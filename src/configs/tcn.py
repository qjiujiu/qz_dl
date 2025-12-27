from src.utils.cuda import select_gpu
from src.configs.utils import with_varients
from src.schemas.context import ExpContext, NetworkConfig, DataConfig, NLPConfig, TrainConfig, AdvConfig
from src.schemas.base_enums import TaskType, OptimizerType, LossType
from src.schemas.block_enums import PluginType
from pathlib import Path

from src.models.tcn_classifer import TCNSeqClassifier


# ==========================================
# MalAPI2019 (单标签多分类)
# ==========================================

nc8 = NetworkConfig(
    name = TCNSeqClassifier.__name__,
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
    description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + No Plugin", 
    network_config = nc8,
    data_config = dc8, 
    train_config = tc,
    adv_config = AdvConfig(
      enable=False
    ),
)


# 各种变体
ctx8_sa, ctx8_sape, ctx8_pe, ctx8_mlp, ctx8_gas = (    
    with_varients(ctx8, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制", 
        plugin=PluginType.SA), 
    with_varients(ctx8, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制 + 位置编码", 
        plugin=PluginType.SelfPE),
    with_varients(ctx8, model_name = TCNSeqClassifier.__name__, 
        suffix="位置编码", 
        plugin=PluginType.PosEnc), 
    with_varients(ctx8, model_name = TCNSeqClassifier.__name__, 
        suffix="MLP注意力机制", 
        plugin=PluginType.MlpAtten), 
    with_varients(ctx8, model_name = TCNSeqClassifier.__name__, 
        suffix="普通高斯噪声", 
        plugin=PluginType.GaussLinf),
)

# 数据版本控制变体 (GC)
ctx_gc = with_varients(ctx8, model_name = TCNSeqClassifier.__name__,  
        suffix="自注意力机制", 
        plugin=PluginType.ID, 
        dataset_name="malapi2019-gc")




# ==========================================
# MalAnalysis (二分类)
# ==========================================

nc2 = NetworkConfig(
    name = TCNSeqClassifier.__name__,
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
    description = f"MalAnalysis 二分类实验: {TCNSeqClassifier.__name__} + No Plugin", 
    network_config = nc2,
    data_config = dc2, 
    train_config = tc, # 复用通用训练配置
    adv_config = AdvConfig(
      enable=False
    ),
)

# 二分类变体
ctx2_sa, ctx2_sape, ctx2_pe, ctx2_mlp, ctx2_gas = (    
    with_varients(ctx2, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制", 
        plugin=PluginType.SA), 
    with_varients(ctx2, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制 + 位置编码", 
        plugin=PluginType.SelfPE),
    with_varients(ctx2, model_name = TCNSeqClassifier.__name__, 
        suffix="位置编码", 
        plugin=PluginType.PosEnc), 
    with_varients(ctx2, model_name = TCNSeqClassifier.__name__, 
        suffix="MLP注意力机制", 
        plugin=PluginType.MlpAtten), 
    with_varients(ctx2, model_name = TCNSeqClassifier.__name__, 
        suffix="普通高斯噪声", 
        plugin=PluginType.GaussLinf),
)




# ==========================================
# MalDynamic (多标签多分类-十五分类)
# ==========================================

ncm = NetworkConfig(
    name = TCNSeqClassifier.__name__,
    dropout_prob = 0.3,
    num_classes=15,
    plugin_type = PluginType.ID,
)

dcm =  DataConfig(
    dataset_name = "maldynamic",
    data_dir = Path("./data/api-calls-generated-by-dynamic-malware-analysis"),
    task_type = TaskType.NLP,
    nlp_config = NLPConfig(
        vocab_size = 332, 
        embedding_dim = 256,
        max_len = 512,
    ),
)

# NOTE 多标签任务必须使用 BCE_LOGITS 损失函数
tcm = TrainConfig(
    batch_size = 512,
    epochs = 20,
    lr = 1e-3,
    optiz = OptimizerType.ADAM,
    device = select_gpu(),
    loss_fn = LossType.BCE_LOGITS, 
)

ctxm = ExpContext(
    description = f"MalDynamic 多标签分类实验: {TCNSeqClassifier.__name__} + No Plugin", 
    network_config = ncm,
    data_config = dcm, 
    train_config = tcm, # 使用专用训练配置
    adv_config = AdvConfig(
      enable=False
    ),
)

# 多标签变体
ctxm_sa, ctxm_sape, ctxm_pe, ctxm_mlp, ctxm_gas = (    
    with_varients(ctxm, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制", 
        plugin=PluginType.SA), 
    with_varients(ctxm, model_name = TCNSeqClassifier.__name__, 
        suffix="自注意力机制 + 位置编码", 
        plugin=PluginType.SelfPE),
    with_varients(ctxm, model_name = TCNSeqClassifier.__name__, 
        suffix="位置编码", 
        plugin=PluginType.PosEnc), 
    with_varients(ctxm, model_name = TCNSeqClassifier.__name__, 
        suffix="MLP注意力机制", 
        plugin=PluginType.MlpAtten), 
    with_varients(ctxm, model_name = TCNSeqClassifier.__name__, 
        suffix="普通高斯噪声", 
        plugin=PluginType.GaussLinf),
)

# ctx = ExpContext(
#     description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + No Plugin", 
#     network_config = NetworkConfig(
#         name = TCNSeqClassifier.__name__,
#         dropout_prob = 0.3,
#         num_classes=8,
#         plugin_type = PluginType.ID,
#     ),
#     data_config = DataConfig(
#         dataset_name = "malapi2019",
#         data_dir = Path("./data/malapi2019"),
#         task_type = TaskType.NLP,
#         nlp_config = NLPConfig(
#             vocab_size = 278, 
#             embedding_dim = 256,
#             max_len = 200,
#         ),
#     ),
#     train_config = TrainConfig(
#         batch_size = 512,
#         epochs = 20,
#         lr = 1e-3,
#         optiz = OptimizerType.ADAM,
#         device = select_gpu(),
#     ),
#     adv_config = AdvConfig(
#       enable=False
#     ),
# )

# ctx_sa = ctx.model_copy(deep=True)
# ctx_sa.description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + 自注意力机制"
# ctx_sa.network_config.plugin_type = PluginType.SA

# ctx_sape = ctx.model_copy(deep=True)
# ctx_sape.description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + 自注意力机制 + 位置编码"
# ctx_sape.network_config.plugin_type = PluginType.SelfPE

# ctx_pe = ctx.model_copy(deep=True)
# ctx_pe.description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + 位置编码"
# ctx_pe.network_config.plugin_type = PluginType.PosEnc

# ctx_mlp = ctx.model_copy(deep=True)
# ctx_mlp.description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + MLP注意力机制"
# ctx_mlp.network_config.plugin_type = PluginType.MlpAtten

# ctx_gas = ctx.model_copy(deep=True)
# ctx_gas.description = f"MalAPI2019 恶意软件API分类实验: {TCNSeqClassifier.__name__} + 普通高斯噪声"
# ctx_gas.network_config.plugin_type = PluginType.GaussLinf