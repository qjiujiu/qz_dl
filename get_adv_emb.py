import os
import torch, pickle
import torch.nn.functional as F


from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.params_template import AdvCfgParams
from config.params_parser.parser import ArgsParser
from config.logger import logger

from tqdm import tqdm
from contextlib import contextmanager
from utils.models import pick_model

@contextmanager
def disable_dropout(model):
    """临时禁用模型 Dropout """
    original_p = model.dropout.p 
    try:
        model.dropout.p = 0.0
        yield  
    finally:
        model.dropout.p = original_p 

class AdversarialAttack:
    def __init__(self, model, cfg: AdvCfgParams):
        self.model = model.to(cfg.device)
        self.fgsm_epsilon = cfg.fgsm_epsilon
        self.pgd_epsilon = cfg.pgd_epsilon
        self.pgd_alpha = cfg.pgd_alpha
        self.pgd_iters = cfg.pgd_iters
        self.device = cfg.device  

    def pgd_attack(self, embed, labels, epsilon = 0.1, alpha=0.01, iters = 5):
        with disable_dropout(self.model):
            embed = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)  

            self.model.train()

            for _ in range(iters):
                self.model.zero_grad()
                output = self.model.forward(adv)

                loss = F.cross_entropy(output, labels)
                loss.backward()

                adv = adv + alpha * adv.grad.sign()
                eta = torch.clamp(adv - embed, -epsilon, epsilon)
                adv = torch.clamp(embed + eta, 0, 1).detach_().requires_grad_(True)  # 更新对抗样本

        return adv.detach()
    
    def fgsm_attack(self, embed, labels, epsilon = 0.1):
        with disable_dropout(self.model):
            embed = embed.clone().detach().to(self.device).requires_grad_(True)
            labels = labels.to(self.device)
        
            self.model.train()
            output = self.model.forward(embed)
            loss = F.cross_entropy(output, labels)
            loss.backward()
    
            adv_emb = embed + epsilon * embed.grad.sign()  # 依据梯度更新对抗样本

        return adv_emb.detach()


# TODO 修改下面两个函数，先把 文本张量变成embedding张量，再把embedding张量丢给攻击算法生成的数据集，生成的数据集不区分训练和测试
def save_pgd_embeddings(model, data_resource, cfg: AdvCfgParams):
    save_dir = "data/malapi2019/emb-feature/LSTMTextClassifier/advexam-pgd"
    os.makedirs(save_dir, exist_ok=True)

    # 获取训练集对抗嵌入
    print("生成训练集的 PGD 对抗嵌入...")
    tot_vectors, tot_labels = [], []
    attacker = AdversarialAttack(model, cfg)
    for texts, labels in tqdm(data_resource.train_loader, desc="Training data", unit="batch"):
        embeds = model.embed(texts)

        # 拆分每个样本，保存为 list of [T, D]
        for embed in embeds.unbind(0):  # unbind along batch dimension
            tot_vectors.append(embed.cpu().detach().numpy())
        
        tot_labels.extend(labels)


    # TODO 下面开始也要这样修改的, 先把所有文本变成向量

    # 保存训练集对抗嵌入
    with open(os.path.join(save_dir, "train_adv_embeddings.pkl"), "wb") as f:
        pickle.dump((tot_vectors, data_resource.y_train), f)

    print("训练集的 PGD 对抗嵌入已保存。")

    # 获取测试集对抗嵌入
    print("生成测试集的 PGD 对抗嵌入...")
    test_adv_embeddings = []
    for texts, tot_labels in tqdm(data_resource.test_loader, desc="Testing data", unit="batch"):
        embeds = AdversarialAttack(model, cfg).pgd_attack(texts, tot_labels)
        # shape: [B, T, D]

        # 拆分每个样本，保存为 list of [T, D]
        for embed in embeds.unbind(0):  # unbind along batch dimension
            test_adv_embeddings.append(embed.cpu().detach().numpy())

    # 保存测试集对抗嵌入
    with open(os.path.join(save_dir, "test_adv_embeddings.pkl"), "wb") as f:
        pickle.dump((test_adv_embeddings, data_resource.y_test), f)

    print("测试集的 PGD 对抗嵌入已保存。")


def save_fgsm_embeddings(model, data_resource, cfg: AdvCfgParams):
    save_dir = "data/malapi2019/emb-feature/LSTMTextClassifier/advexam-fgsm"
    os.makedirs(save_dir, exist_ok=True)

    # 获取训练集对抗嵌入
    print("生成训练集的 FGSM 对抗嵌入...")
    train_adv_embeddings = []
    for texts, labels in tqdm(data_resource.train_loader, desc="Training data", unit="batch"):
        adv_embeds = AdversarialAttack(model, cfg).fgsm_attack(texts, labels)
        # shape: [B, T, D]
        # 拆分每个样本，存为 list of [T, D]
        for embed in adv_embeds.unbind(0):  # unbind along batch dimension
            train_adv_embeddings.append(embed.cpu().detach().numpy())
    with open(os.path.join(save_dir, "train_adv_embeddings.pkl"), "wb") as f:
        pickle.dump((train_adv_embeddings, data_resource.y_train), f)

    print("训练集的 FGSM 对抗嵌入已保存。")

    # 获取测试集对抗嵌入
    print("生成测试集的 FGSM 对抗嵌入...")
    test_adv_embeddings = []
    for texts, labels in tqdm(data_resource.test_loader, desc="Testing data", unit="batch"):
        adv_embeds = AdversarialAttack(model, cfg).fgsm_attack(texts, labels)
        # shape: [B, T, D]

        # 拆分每个样本，保存为 list of [T, D]
        for embed in adv_embeds.unbind(0):  # unbind along batch dimension
            test_adv_embeddings.append(embed.cpu().detach().numpy())

    with open(os.path.join(save_dir, "test_adv_embeddings.pkl"), "wb") as f:
        pickle.dump((test_adv_embeddings, data_resource.y_test), f)

    print("测试集的 FGSM 对抗嵌入已保存。")
 


"""
chenzc: 
    python get_adv_emb.py --model lstm --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8 --max-len 200 --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-1304-46c8ff3d_weights.pth
""" 
if __name__ == "__main__":
    # 加载配置和数据
    cfg = ArgsParser().create_adv_config()
    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi", 
        batch_size=cfg.batch_size, 
    )
    # 加载预训练的模型
    model = pick_model(cfg, cfg.checkpoint_path)

    # 保存对抗嵌入
    save_fgsm_embeddings(model, data_resource, cfg)
    save_pgd_embeddings(model, data_resource, cfg)
    
