import torch
import pickle
import os
from tqdm import tqdm
from config.params_parser.parser import ArgsParser
from config.logger import logger
from utils.models import pick_model
from config.params_parser.params_template import NlpCfgParams
from config.datasets.datasrc.text_datasrc import TextDataSrc
from utils import io

def save_embeddings(embeddings, labels, save_dir, file_name):
    """保存嵌入表示"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump((embeddings, labels), f)
    logger.debug(f"嵌入表示已保存到 {save_path}")


def get_embeding(cfg: NlpCfgParams):
    # 文件路径
    cache_dir = "data/malapi2019/preprocessed"
    save_dir = "data/malapi2019/emb-feature/LSTMTextClassifier/clean-exam"
    
    file_names = ['train_texts.pkl', 'test_texts.pkl', 'train_labels.pkl', 'test_labels.pkl', 'vocab.pkl']
    cache_files_exist = all(os.path.exists(os.path.join(cache_dir, file)) for file in file_names)

    if cache_files_exist:
        logger.debug("📦 正在缓存 malapi_cleanexm 干净嵌入数据集...")

        data_resource = TextDataSrc.load_dataset(
            dataset_name="malapi", 
            batch_size=cfg.batch_size, 
        )

        device = torch.device(cfg.device) 

        # 加载权重模型
        model = pick_model(cfg, cfg.checkpoint_path)
        model = model.to(device)

        print("生成训练集的干净嵌入...")
        train_embeddings = []
        for texts, labels in tqdm(data_resource.train_loader, desc="Training data", unit="batch"):
            texts = texts.to(device)
            embedded = model.embed(texts)
            # shape: [B, T, D]

            # 拆分每个样本，保存为 list of [T, D]
            for embed in embedded.unbind(0):  # unbind along batch dimension
                train_embeddings.append(embed.cpu().detach().numpy())

        with open(os.path.join(save_dir, "train_embeddings.pkl"), "wb") as f:
            pickle.dump((train_embeddings, data_resource.y_train), f)

        print("生成测试集的干净嵌入...")
        test_embeddings = []
        for texts, labels in tqdm(data_resource.test_loader, desc="Testing data", unit="batch"):
            texts = texts.to(device)
            embedded = model.embed(texts)
            # shape: [B, T, D]

            # 拆分每个样本，保存为 list of [T, D]
            for embed in embedded.unbind(0):  # unbind along batch dimension
                test_embeddings.append(embed.cpu().detach().numpy())

        with open(os.path.join(save_dir, "test_embeddings.pkl"), "wb") as f:
            pickle.dump((test_embeddings, data_resource.y_test), f)

    else:
        logger.debug("📦 原数据不存在在该路径")
        return


# python mal_api_embed.py --checkpoint-path checkpoints/2025-07-07/LSTMTextClassifier/20250707-1933-e1404e59_weights.pth --model lstm  --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
if __name__ == "__main__":
    cfg = ArgsParser().create_nlp_config()
    get_embeding(cfg)