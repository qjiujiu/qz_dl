# train_ml_corrected.py
import argparse
import importlib
import numpy as np
import torch
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.multioutput import MultiOutputClassifier # <--- 关键：多标签包装器

# 尝试导入 boosting 模型
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from src.utils.logx import logger
from src.schemas.context import ExpContext
from src.schemas.base_enums import LossType
from src.datasets.instances import malanalysis, malapi2019, maldynamic

# 释放缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()


def load_config_ctx(config_name: str, ctx_name: str, args: argparse.Namespace = None) -> ExpContext:
    module_path = f"src.configs.{config_name}"
    try:
        config_module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ValueError(f"Config module not found: '{module_path}'") from e

    if not hasattr(config_module, ctx_name):
        raise AttributeError(f"Module `{module_path}` does not define `{ctx_name}`!")
    
    ctx_obj: ExpContext = getattr(config_module, ctx_name)
    logger.success(f"Loaded config: `src.configs.{config_name}.{ctx_name}`")
    return ctx_obj

def get_data_loader_func(dataset_name: str):
    if dataset_name.lower() == 'malapi2019':
        return malapi2019
    elif dataset_name.lower() == 'malanalysis':
        return malanalysis
    elif dataset_name.lower() == 'maldynamic':
        return maldynamic
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def pytorch_loader_to_sklearn(loader, is_multilabel=False, desc="Converting"):
    """
    将 PyTorch DataLoader 转换为 Sklearn 格式。
    Args:
        is_multilabel: 如果为 True，保留 y 为 [N, C] 的 0/1 矩阵；否则转为 1D 索引。
    """
    X_data = []
    y_data = []
    
    for x, y in tqdm(loader, desc=desc):
        # x: [batch, seq_len], y: [batch, num_classes]
        
        # 1. 处理标签 (关键修复)
        if is_multilabel:
            # 多标签：直接转为 int 类型的 0/1 矩阵
            # 如果 y 是 logits 或 float，先转 int
            y_np = y.cpu().numpy().astype(int)
            y_data.extend(y_np)
        else:
            # 单标签多分类：如果 y 还是 One-hot 的，转为 index
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            y_data.extend(y.cpu().numpy())

        # 2. 处理特征 (API 序列 -> 字符串)
        x_np = x.cpu().numpy()
        for seq in x_np:
            seq_str = " ".join([str(idx) for idx in seq if idx != 0])
            X_data.append(seq_str)
            
    return X_data, np.array(y_data)

def train_and_evaluate(model, model_name, X_train_vec, y_train, X_test_vec, y_test, is_multilabel=False):
    if model is None:
        logger.warning(f"⚠️  Skipping {model_name} (Library not installed)")
        return

    logger.info(f"🚀 Training {model_name}...")

    # =======================================================
    # 【关键修复】正确判断是否需要 MultiOutputClassifier 包装
    # =======================================================
    if is_multilabel:
        needs_wrapping = False
        
        # 1. 检查 LinearSVC
        if isinstance(model, LinearSVC):
            needs_wrapping = True
            
        # 2. 检查 XGBoost (如果库存在)
        elif XGBClassifier is not None and isinstance(model, XGBClassifier):
            needs_wrapping = True
            
        # 3. 检查 LightGBM (如果库存在)
        elif LGBMClassifier is not None and isinstance(model, LGBMClassifier):
            needs_wrapping = True

        if needs_wrapping:
            logger.info(f"   -> Wrapping {model_name} with MultiOutputClassifier for Multi-label support.")
            # n_jobs=-1 让包装器并行训练多个分类器，加速明显
            model = MultiOutputClassifier(model, n_jobs=-1)

    try:
        model.fit(X_train_vec, y_train)
    except ValueError as e:
        logger.error(f"❌ Training failed for {model_name}: {e}")
        # 打印一下形状帮助调试
        logger.error(f"   Debug Info: X shape={X_train_vec.shape}, y shape={y_train.shape}")
        return
    
    logger.info(f"🔍 Predicting {model_name}...")
    y_pred = model.predict(X_test_vec)
    
    # 评估指标
    if is_multilabel:
        acc = accuracy_score(y_test, y_pred)
        micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        logger.success(f"✅ {model_name} | Subset Acc: {acc:.4f} | Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f}")
    else:
        acc = accuracy_score(y_test, y_pred)
        logger.success(f"✅ {model_name} Accuracy: {acc:.4f}")
    
    try:
        # target_names 能够让 report 更易读，如果你有 labels 列表可以传进去
        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        print(report)
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Baselines")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config module name")
    parser.add_argument("-n", "--name", type=str, default="ctx", help="Ctx variable name")
    parser.add_argument("-d", "--dataset", type=str, default="malapi2019", help="Dataset name")
    parser.add_argument("--max-feature", type=int, default=10000)
    parser.add_argument("--ngram", type=int, default=1)
    
    args = parser.parse_args()
    ctx = load_config_ctx(args.config, args.name, args)
    
    # 1. 自动判断任务类型 (Multi-label vs Multi-class)
    # 依据 loss_fn 或 dataset 名字判断
    is_multilabel = False
    if ctx.train_config.loss_fn == LossType.BCE_LOGITS:
        is_multilabel = True
        logger.info("⚡ Detected Multi-label Task (BCE Loss).")
    else:
        logger.info("⚡ Detected Multi-class Task (CrossEntropy Loss).")

    # 2. 加载数据
    data_module = get_data_loader_func(args.dataset)
    train_loader, val_loader = data_module.build_datamodule(ctx) # 你的函数可能返回 vocab，这里忽略
    
    # 3. 转换数据
    logger.info("Converting PyTorch tensors to TF-IDF compatible format...")
    # 传入 is_multilabel 标志，决定是否保留矩阵形式
    X_train_raw, y_train = pytorch_loader_to_sklearn(train_loader, is_multilabel=is_multilabel, desc="Train Data")
    X_test_raw, y_test = pytorch_loader_to_sklearn(val_loader, is_multilabel=is_multilabel, desc="Test Data")
    
    logger.info(f"Label Shape: {y_train.shape}")
    if is_multilabel:
        logger.info(f"Example Label (First 2): \n{y_train[:2]}")

    # 4. 特征提取
    logger.info(f"Vectorizing data (Max Features: {args.max_feature}, N-gram: 1-{args.ngram})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, args.ngram), 
        max_features=args.max_feature,
        token_pattern=r"(?u)\b\w+\b"
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)
    
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ("Linear SVM", LinearSVC()),
        ("LightGBM", LGBMClassifier()),
    ]


    # 6. 训练
    print("=" * 60)
    for name, model in models:
        train_and_evaluate(model, name, X_train_vec, y_train, X_test_vec, y_test, is_multilabel=is_multilabel)