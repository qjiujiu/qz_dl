import torch
import pickle
import os
from tqdm import tqdm
from config.params_parser.parser import ArgsParser
from config.logger import logger
from utils.models import pick_model
from config.datasets.dataset_instance import mal_api

def load_model(model_instance, checkpoint_path):
    """加载已训练的模型"""
    # 加载模型权重
    try:
        model_instance.load_state_dict(torch.load(checkpoint_path))
        logger.debug(f"模型权重从 {checkpoint_path} 成功加载")
    except Exception as e:
        logger.error(f"加载模型权重失败: {e}")
        raise
    
    model_instance.eval()  # 切换为评估模式
    return model_instance


def save_embeddings(embeddings, labels, save_dir, file_name):
    """保存嵌入表示"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump((embeddings, labels), f)
    logger.debug(f"嵌入表示已保存到 {save_path}")


def get_embeding(cfg):
    # 文件路径
    data_dir = "data/malapi2019/preprocessed"
    save_dir = "data/malapi2019/emb-feature/LSTMTextClassifier/clean-exam"

    # 检查是否已有缓存的嵌入
    if os.path.exists(os.path.join(save_dir, "train_embeddings.pkl")) and os.path.exists(os.path.join(save_dir, "test_embeddings.pkl")):
        logger.debug("嵌入表示缓存文件已存在，直接加载缓存...")
        return  # 缓存已存在，直接返回

    # 加载文本数据和标签
    try:
        with open(os.path.join(data_dir, "train_texts.pkl"), "rb") as f:
            train_texts = pickle.load(f)
        with open(os.path.join(data_dir, "test_texts.pkl"), "rb") as f:
            test_texts = pickle.load(f)
        with open(os.path.join(data_dir, "train_labels.pkl"), "rb") as f:
            train_labels = pickle.load(f)
        with open(os.path.join(data_dir, "test_labels.pkl"), "rb") as f:
            test_labels = pickle.load(f)
        with open(os.path.join(data_dir, "vocab.pkl"), "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"文件加载失败: {e}")
        return
    
    logger.debug(f"加载了训练集和测试集样本，共 {len(train_texts)} 个训练样本和 {len(test_texts)} 个测试样本")

    device = torch.device(cfg.device) 

    # 加载模型
    model_instance = pick_model(cfg)
    model = load_model(model_instance, cfg.checkpoint_path)
    model = model.to(device)

    train_dataset = mal_api.MalAPITextDataset(texts=train_texts, labels=train_labels, vocab=vocab)
    test_dataset = mal_api.MalAPITextDataset(texts=test_texts, labels=test_labels, vocab=vocab)

    # 获取嵌入表示并保存
    logger.debug("获取训练集嵌入表示...")
    train_embeddings = []
    for texts, labels in tqdm(train_dataset, desc="Processing train texts"):
        # 假设文本已被填充到正确的大小，直接传入模型
        texts = texts.to(device)
        embedded = model.embed(texts)  # [batch_size, max_len] 格式
        train_embeddings.extend(embedded.cpu().detach().numpy())
        
    save_embeddings(train_embeddings, train_labels, save_dir, "train_embeddings.pkl")
    
    logger.debug("获取测试集嵌入表示...")
    test_embeddings = []
    for texts, labels in tqdm(test_dataset, desc="Processing test texts"):
        texts = texts.to(device)
        embedded = model.embed(texts)  # [batch_size, max_len] 格式
        test_embeddings.extend(embedded.cpu().detach().numpy())
    
    
    save_embeddings(test_embeddings, test_labels, save_dir, "test_embeddings.pkl")

# python mal_api_embed.py --checkpoint-path checkpoints/2025-07-07/LSTMTextClassifier/20250707-1933-e1404e59_weights.pth --model lstm  --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
if __name__ == "__main__":
    cfg = ArgsParser().create_nlp_config()
    get_embeding(cfg)