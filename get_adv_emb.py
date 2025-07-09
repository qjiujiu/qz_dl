import torch
import torch.nn.functional as F
import pickle
import os
from config.params_parser.parser import ArgsParser
from config.logger import logger
from utils.models import pick_model
from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.params_template import AdvCfgParams
from tqdm import tqdm

class AdversarialAttack:
    def __init__(self, model, cfg: AdvCfgParams):
        self.model = model
        self.fgsm_epsilon = cfg.fgsm_epsilon
        self.pgd_epsilon = cfg.pgd_epsilon
        self.pgd_alpha = cfg.pgd_alpha
        self.pgd_iters = cfg.pgd_iters
        self.device = cfg.device  
        self.model.to(self.device)

    def pgd_attack(self, texts, labels):
        """
        生成 PGD 对抗样本 (针对模型的 Embedding 层的输出)
        """
        # 设置模型为训练模式，以支持反向传播
        self.model.train()

        # 临时禁用 dropout
        original_dropout_p = self.model.dropout.p
        self.model.dropout.p = 0.0

        # 将输入数据和标签传入设备
        texts = texts.to(self.device)  
        labels = labels.to(self.device)  

        # 获取初始嵌入表示
        adv = self.model.embed(texts).detach().requires_grad_(True)  # 初始化对抗样本
        ori = adv.detach()

        for _ in range(self.pgd_iters):
            self.model.zero_grad()
            # 使用当前的对抗样本进行前向传播
            output = self.model.forward(adv)
            # 计算交叉熵损失
            loss = F.cross_entropy(output, labels)
            loss.backward()
            # 计算对抗扰动
            adv = adv + self.pgd_alpha * adv.grad.sign()  # 依据梯度更新对抗样本
            eta = torch.clamp(adv - ori, -self.pgd_epsilon, self.pgd_epsilon)  # 限制扰动范围
            adv = torch.clamp(ori + eta, 0, 1).detach_().requires_grad_(True)  # 更新对抗样本

        # 恢复模型的 dropout 设置
        self.model.dropout.p = original_dropout_p
        return adv.detach()
    
    def fgsm_attack(self, texts, labels):
        """
        生成 FGSM 对抗样本 (针对模型的 Embedding 层的输出)
        """
        # 设置模型为训练模式，以支持反向传播
        self.model.train()

        # 临时禁用 dropout
        original_dropout_p = self.model.dropout.p
        self.model.dropout.p = 0.0

        # 将输入数据和标签传入设备
        texts = texts.to(self.device)
        labels = labels.to(self.device)

        # 获取初始嵌入表示
        adv = self.model.embed(texts).detach().requires_grad_(True)  # 初始化对抗样本
        # 前向传播
        output = self.model.forward(adv)
        # 计算损失并反向传播
        loss = F.cross_entropy(output, labels)
        loss.backward()
        # 生成对抗样本
        adv_emb = adv + self.fgsm_epsilon * adv.grad.sign()  # 依据梯度更新对抗样本

        # 恢复模型的 dropout 设置
        self.model.dropout.p = original_dropout_p
        return adv_emb.detach()

def save_pgd_embeddings(model, data_resource, cfg: AdvCfgParams):
    save_dir = "data/malapi2019/emb-feature/LSTMTextClassifier/advexam-pgd"
    os.makedirs(save_dir, exist_ok=True)

    # 获取训练集对抗嵌入
    print("生成训练集的 PGD 对抗嵌入...")
    train_adv_embeddings = []
    for texts, labels in tqdm(data_resource.train_loader, desc="Training data", unit="batch"):
        adv_embeds = AdversarialAttack(model, cfg).pgd_attack(texts, labels)
        # shape: [B, T, D]

        # 拆分每个样本，保存为 list of [T, D]
        for embed in adv_embeds.unbind(0):  # unbind along batch dimension
            train_adv_embeddings.append(embed.cpu().detach().numpy())

    # 保存训练集对抗嵌入
    with open(os.path.join(save_dir, "train_adv_embeddings.pkl"), "wb") as f:
        pickle.dump((train_adv_embeddings, data_resource.y_train), f)

    print("训练集的 PGD 对抗嵌入已保存。")

    # 获取测试集对抗嵌入
    print("生成测试集的 PGD 对抗嵌入...")
    test_adv_embeddings = []
    for texts, labels in tqdm(data_resource.test_loader, desc="Testing data", unit="batch"):
        adv_embeds = AdversarialAttack(model, cfg).pgd_attack(texts, labels)
        # shape: [B, T, D]

        # 拆分每个样本，保存为 list of [T, D]
        for embed in adv_embeds.unbind(0):  # unbind along batch dimension
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
 



# python get_adv_emb.py --model lstm --lr 0.001 -eb 128 --hidden-dim 256 --output-dim 8 --max-len 200 --vocab-size 278 -cp checkpoints/2025-07-09/LSTMTextClassifier/20250709-0954-ff28631f_weights.pth
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
    
