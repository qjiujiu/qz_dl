import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.dataset_instance.mal_api import load
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from config.logger import logger_initiate
from tqdm import tqdm
import os
from utils.get_config import load_config

# FGSM 攻击
def fgsm_attack(model, emb_input, labels, epsilon, device):
    emb_input = emb_input.clone().detach().to(device).requires_grad_(True)
    labels = labels.to(device)
    # torch.nn.LSTM 的 cuDNN 实现在 eval() 模式下 禁止反向传播
    # model.eval()
    
    # 设置为 train 模式以支持 RNN 反向传播
    model.train()
    # 临时禁用 dropout
    original_dropout_p = model.dropout.p
    model.dropout.p = 0.0
    
    output = model.forward(emb_input)
    loss = F.cross_entropy(output, labels)
    loss.backward()
    adv_emb = emb_input + epsilon * emb_input.grad.sign()
    return adv_emb.detach()

# PGD 攻击
def pgd_attack(model, emb_input, labels, epsilon, alpha, iters, device):
    ori = emb_input.clone().detach().to(device)
    adv = ori.clone().detach().requires_grad_(True)
    labels = labels.to(device)
    # torch.nn.LSTM 的 cuDNN 实现在 eval() 模式下 禁止反向传播
    # model.eval()

    # 设置为 train 模式以支持 RNN 反向传播
    model.train()
    # 临时禁用 dropout
    original_dropout_p = model.dropout.p
    model.dropout.p = 0.0

    for _ in range(iters):
        model.zero_grad()
        output = model.forward(adv)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        eta = torch.clamp(adv - ori, -epsilon, epsilon)
        adv = torch.clamp(ori + eta, 0, 1).detach_().requires_grad_(True)
    return adv.detach()

# 缓存嵌入表示
def cache_embeddings(model, dataloader, device, cache_path='data/malapi2019/emb-feature/clean-exam'):
    clean_path = os.path.join(cache_path, 'clean_examples.pt')
    if os.path.exists(clean_path):
        return torch.load(clean_path)
    
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Caching embeddings"):
            texts = texts.to(device)
            labels = labels.to(device)
            emb = model.embedding(texts)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels.cpu())

    cached_emb = torch.cat(all_embeddings)
    cached_labels = torch.cat(all_labels)

    os.makedirs(cache_path, exist_ok=True)
    torch.save((cached_emb, cached_labels), clean_path)

    return cached_emb, cached_labels

    # FGSM 样本生成
def generate_fgsm_examples(config, model, cached_emb, cached_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = config['fgsm_epsilon']

    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))

    logger = logger_initiate(log_level='INFO', is_console=True, is_file=True, is_colorful=True)
    adv_fgsm = fgsm_attack(model, cached_emb, cached_labels, epsilon, device)
    os.makedirs('data/malapi2019/emb-feature/advexam-fgsm', exist_ok=True)
    torch.save((adv_fgsm.cpu(), cached_labels), 'data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt')
    logger.info("FGSM adversarial examples saved.")

# PGD 样本生成
def generate_pgd_examples(config, model, cached_emb, cached_labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = config['pgd_epsilon']
    alpha = config['pgd_alpha']
    pgd_iters = config['pgd_iters']

    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))

    logger = logger_initiate(log_level='INFO', is_console=True, is_file=True, is_colorful=True)
    adv_pgd = pgd_attack(model, cached_emb, cached_labels, epsilon, alpha, pgd_iters, device)
    os.makedirs('data/malapi2019/emb-feature/advexam-pgd', exist_ok=True)
    torch.save((adv_pgd.cpu(), cached_labels), 'data/malapi2019/emb-feature/advexam-pgd/pgd.pt')
    logger.info("PGD adversarial examples saved.")


def get_adv_emb_normal(config, model):
    # 加载数据集
    full_dataset, vocab = load(config['train_data'], config['train_labels'], split=False)
    full_loader = DataLoader(full_dataset, batch_size=config['batch_size'], shuffle=False)


    cached_emb, cached_labels = cache_embeddings(model, full_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 生成 FGSM 和 PGD 样本
    generate_fgsm_examples(config, model, cached_emb, cached_labels)
    generate_pgd_examples(config, model, cached_emb, cached_labels)

def get_adv_emb_MiniLML6(config, model, example_path = 'data/malapi2019/emb-MiniLM-L6/clean_examples.pt'):
    # 加载干净样本和标签
    cached_emb, cached_labels = torch.load(example_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epsilon = config['fgsm_epsilon']  # FGSM epsilon
    alpha = config['pgd_alpha']  # PGD alpha
    pgd_iters = config['pgd_iters']  # PGD iterations

    # 生成 FGSM 对抗样本
    adv_fgsm = fgsm_attack(model, cached_emb, cached_labels, epsilon, device)
    fgsm_save_path = os.path.join(os.path.dirname(example_path), 'adv_fgsm.pt')
    torch.save((adv_fgsm.cpu(), cached_labels), fgsm_save_path)

    # 生成 PGD 对抗样本
    adv_pgd = pgd_attack(model, cached_emb, cached_labels, epsilon, alpha, pgd_iters, device)
    pgd_save_path = os.path.join(os.path.dirname(example_path), 'adv_pgd.pt')
    torch.save((adv_pgd.cpu(), cached_labels), pgd_save_path)

    print(f"FGSM adversarial examples saved to {fgsm_save_path}")
    print(f"PGD adversarial examples saved to {pgd_save_path}")

if __name__ == "__main__":
    config = load_config("config/lstm_config.yaml")  # 加载配置
    # 加载模型并缓存嵌入
    model = LSTMTextClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    

    # 在普通嵌入层得到的Embedding特征上生成对抗样本
    # get_adv_emb_normal(config, model)

    # 从 MiNiLM 模型得到的Embedding特征上生成对抗样本
    get_adv_emb_MiniLML6(config, model)