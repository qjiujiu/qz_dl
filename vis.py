import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path

# 引入你的项目模块
from src.utils.logx import logger, logging

from src.datasets.instances import malapi2019
from src.models.build_model import build_model


import importlib
from src.schemas.context import ExpContext
from src.schemas.base_enums import LossType


logger.setLevel(logging.INFO)

def load_config_ctx(config_name: str, ctx_name: str) -> ExpContext:
    module_path = f"src.configs.{config_name}"
    config_module = importlib.import_module(module_path)
    return getattr(config_module, ctx_name)

def load_checkpoint(model, checkpoint_path):
    """ 加载训练好的权重 """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 兼容处理：有些 checkpoint 保存的是 {'model_state_dict': ...}，有些直接是 state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ==========================================
# 2. 核心逻辑：生成对抗样本
# ==========================================
def get_adv_embeddings(ctx, model, x, y, device):
    """
    对输入 x 进行 PGD 攻击，返回 (Clean Embeddings, Adv Embeddings, Delta)
    """
    # 1. 获取 Clean Embedding
    # 获取 clean embedding 时可以用 eval 模式
    model.eval() 
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        clean_emb = model.embedding(x)
    
    # 2. 准备 PGD 攻击
    eps = 0.3
    steps = ctx.adv_config.steps
    alpha = 0.1
    
    delta = torch.randn_like(clean_emb).to(device) * (eps / 2)
    delta.data.normal_(0, eps / 2)
    delta.data = torch.clamp(delta.data, -eps, eps)
    delta.requires_grad = True
    
    loss_fn = nn.CrossEntropyLoss()
    model.train() 

    # 攻击循环 (生成噪声)
    with torch.enable_grad():
        for _ in range(steps):
            outputs = model(x, embed_perturbation=delta)
            loss = loss_fn(outputs, y)
            loss.backward() # 此时 model.train() 为 True，不会报错
            
            if delta.grad is not None:
                g = delta.grad.detach()
                delta.data = delta.data + alpha * torch.sign(g)
                delta.data = torch.clamp(delta.data, -eps, eps)
                delta.grad.zero_()
    
    # ==========================================
    # [关键修复] 攻击结束，切回 Eval 模式
    # ==========================================
    model.eval() 
    
    # 计算最终的对抗 Embedding (仅 Forward，不需要梯度)
    # 这一步是为了可视化，应该在 Eval 模式下拿最终结果
    adv_emb = clean_emb + delta.detach()
    # ==========================================
    # 4. 结果整理与 Rank 计算
    # ==========================================
    model.eval() 
    
    # 计算最终的对抗 Embedding
    adv_emb = clean_emb + delta.detach()
    
    # [TODO 完成] 计算并打印矩阵秩 (Rank)
    # clean_emb shape: [Batch, Seq_Len, Dim]
    # torch.linalg.matrix_rank 会自动对 Batch 维度进行广播计算，返回 [Batch] 大小的秩向量
    with torch.no_grad():
        # 计算整个 Batch 的秩
        c_ranks = torch.linalg.matrix_rank(clean_emb)
        a_ranks = torch.linalg.matrix_rank(adv_emb)
        
        # 计算平均秩 (转为 float 计算均值)
        avg_c_rank = c_ranks.float().mean().item()
        avg_a_rank = a_ranks.float().mean().item()
        
        # 也可以计算一下稀疏度 (Sparsity)，即 0 元素的比例，看是否变“稠密”了
        # 这里假设绝对值小于 1e-5 视为 0
        c_sparsity = (clean_emb.abs() < 1e-5).float().mean().item()
        a_sparsity = (adv_emb.abs() < 1e-5).float().mean().item()

        logger.info("-" * 40)
        logger.info(f"📊 Matrix Statistics (Batch Avg):")
        logger.info(f"   Shape: {clean_emb.shape[1:]} (Seq, Dim)")
        logger.info(f"   Clean Rank: {avg_c_rank:.2f} | Sparsity: {c_sparsity:.2%}")
        logger.info(f"   Adv   Rank: {avg_a_rank:.2f} | Sparsity: {a_sparsity:.2%}")
        logger.info(f"   -> Rank Delta: {avg_a_rank - avg_c_rank:+.2f}")
        logger.info("-" * 40)

    return clean_emb.detach(), adv_emb.detach(), delta.detach()


# ==========================================
# 3. 可视化功能 A: 散点图 (t-SNE/PCA)
# ==========================================
def visualize_distribution(clean_emb, adv_emb, labels, save_path="vis_distribution.png"):
    """
    将 [Batch, Seq, Dim] 的 Embedding 降维并可视化
    为了可视化方便，我们先对 Seq 维度做 Mean Pooling -> [Batch, Dim]
    """
    logger.info("🎨 Plotting Distribution (t-SNE)...")
    
    # 1. Pooling: [B, L, D] -> [B, D]
    clean_vec = clean_emb.mean(dim=1).cpu().numpy()
    adv_vec = adv_emb.mean(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 如果是多标签，取第一个标签用于着色，或者只画点
    if len(labels.shape) > 1: 
        labels = labels.argmax(axis=1)

    # 2. 合并数据进行 t-SNE (保证空间一致性)
    batch_size = clean_vec.shape[0]
    data = np.concatenate([clean_vec, adv_vec], axis=0) # [2B, D]
    
    # 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, batch_size//2))
    # 或者用 PCA: pca = PCA(n_components=2); X_2d = pca.fit_transform(data)
    X_2d = tsne.fit_transform(data)
    
    clean_2d = X_2d[:batch_size]
    adv_2d = X_2d[batch_size:]
    
    # 3. 绘图
    plt.figure(figsize=(10, 8))
    
    # 画 Clean 点 (圆形)
    sns.scatterplot(x=clean_2d[:,0], y=clean_2d[:,1], hue=labels, palette="deep", 
                    style=["Clean"]*batch_size, markers=["o"], s=100, legend='brief')
    
    # 画 Adv 点 (叉号)
    # 注意：为了让颜色对应，我们手动循环画或者简单处理
    sns.scatterplot(x=adv_2d[:,0], y=adv_2d[:,1], hue=labels, palette="deep", 
                    style=["Adv"]*batch_size, markers=["X"], s=100, legend=False)
    
    # 画连接线 (显示移动轨迹)
    for i in range(batch_size):
        plt.arrow(clean_2d[i,0], clean_2d[i,1], 
                  adv_2d[i,0]-clean_2d[i,0], adv_2d[i,1]-clean_2d[i,1], 
                  color='gray', alpha=0.3, width=0.002)

    plt.title("Embedding Shift: Clean vs Adversarial (t-SNE)")
    plt.savefig(save_path, dpi=300)
    logger.success(f"Saved distribution plot to {save_path}")
    plt.close()



# ==========================================
# 4. 可视化功能 B: 矩阵热力图 (平滑版)
# ==========================================
import matplotlib.pyplot as plt
import numpy as np
from src.utils.logx import logger



import matplotlib.pyplot as plt
import numpy as np
from src.utils.logx import logger

# ==========================================
# 4. 可视化功能 B: 矩阵热力图 (平滑版 - 全红蓝配色)
# ==========================================
def visualize_heatmap_vertical(clean_emb, delta, adv_emb, sample_idx=0, save_path="vis_heatmap_vertical.pdf", theme='academic'):
    logger.info(f"🎨 Plotting Minimalist Vertical Heatmaps to PDF...")
    
    # 1. 取数据 & 2. 截断
    c = clean_emb[sample_idx].cpu().numpy()
    d = delta[sample_idx].cpu().numpy()
    a = adv_emb[sample_idx].cpu().numpy()
    
    # 逻辑修正：确保数据纯净，不重复叠加
    # a += d  <-- 已删除
    
    time_steps_to_show = 40
    dims_to_show = 25
    
    c = c[:time_steps_to_show, :dims_to_show]
    d = d[:time_steps_to_show, :dims_to_show]
    a = a[:time_steps_to_show, :dims_to_show]
    
    # =======================================================
    # 【配置】统一配色方案 & Scale
    # =======================================================
    target_cmap = 'RdBu_r' 
    delta_abs_max = np.max(np.abs(d))
    emb_abs_max = max(np.max(np.abs(c)), np.max(np.abs(a)))
    interpolation_mode = 'bicubic' 
    
    # 3. 创建画布
    fig, axes = plt.subplots(1, 3, figsize=(8, 5))
    
    # 定义添加 Colorbar 的辅助函数
    def add_colorbar(im, ax):
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.05, pad=0.08, aspect=20)
        cbar.ax.tick_params(labelsize=8)

    # =======================================================
    # 定义通用样式函数 (DRY原则)
    # =======================================================
    def style_ax(ax, title):
        ax.set_title(title, fontsize=9, fontweight='bold', pad=8)
        # 【关键修改】隐藏 X 轴刻度
        ax.set_xticks([]) 
        # 【关键修改】让 Dim 贴得更紧 (labelpad=2)
        ax.set_xlabel("Dim", fontsize=10, labelpad=2)
        # 隐藏 Y 轴刻度 (默认)
        ax.set_yticks([])
        ax.set_ylabel("")

    # --- (a) Clean ---
    im0 = axes[0].imshow(c, cmap=target_cmap, interpolation=interpolation_mode, aspect='auto',
                         vmin=-emb_abs_max, vmax=emb_abs_max)
    add_colorbar(im0, axes[0])
    
    # 应用样式
    style_ax(axes[0], "(a) Clean Representation")
    # (a) 图需要保留 Y 轴标签 (Time Step)
    axes[0].set_ylabel("Time Step", fontsize=10, labelpad=2)
    # 恢复 Y 轴刻度显示 (因为 style_ax 默认关掉了)
    # 如果想保留 Y 轴数字：
    # axes[0].set_yticks(np.arange(0, time_steps_to_show, 10))
    # axes[0].set_yticklabels(np.arange(0, time_steps_to_show, 10))
    # 如果只想保留 "Time Step" 文字但不要数字，就保持 set_yticks([])
    # 这里假设你保留左侧的数字刻度以便读者理解时间流逝：
    axes[0].set_yticks(np.linspace(0, time_steps_to_show-1, 5))
    axes[0].set_yticklabels(np.linspace(0, time_steps_to_show, 5, dtype=int))

    # --- (b) Delta ---
    im1 = axes[1].imshow(d, cmap=target_cmap, interpolation=interpolation_mode, aspect='auto',
                         vmin=-delta_abs_max, vmax=delta_abs_max)
    add_colorbar(im1, axes[1])
    style_ax(axes[1], "(b) Signal provided by NIM")

    # --- (c) Adv ---
    im2 = axes[2].imshow(a, cmap=target_cmap, interpolation=interpolation_mode, aspect='auto',
                         vmin=-emb_abs_max, vmax=emb_abs_max)
    add_colorbar(im2, axes[2])
    style_ax(axes[2], "(c) Augument Representation")
    
    # 4. 调整布局
    plt.subplots_adjust(wspace=0.1, bottom=0.15) # bottom 稍微改小一点，因为去掉了刻度
    
    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    logger.success(f"Saved minimalist vector PDF heatmap to {save_path}")
    plt.close()
    
    
# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Embeddings & Adversarial Noise")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config name (e.g., lstm)")
    parser.add_argument("-n", "--name", type=str, default="ctx", help="Ctx name")
    parser.add_argument("-p", "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of samples to visualize")
    
    # 可以临时修改攻击参数来观察不同强度的噪声
    parser.add_argument("--epsilon", type=float, default=None)
    
    args = parser.parse_args()
    
    # 1. 加载配置
    ctx = load_config_ctx(args.config, args.name)
    if args.epsilon:
        ctx.adv_config.epsilon = args.epsilon
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载模型 & 权重
    model = build_model(ctx)
    model = load_checkpoint(model, args.checkpoint)
    model.to(device)
    
    # 3. 加载数据 (只取验证集的一个 Batch)
    _, val_loader, _ = malapi2019.build_datamodule(ctx)
    x, y = next(iter(val_loader))
    
    # 截取指定数量样本
    x = x[:args.batch_size]
    y = y[:args.batch_size]
    
    logger.info(f"Visualizing batch of shape: {x.shape}")
    
    # 4. 生成数据
    clean, adv, delta = get_adv_embeddings(ctx, model, x, y, device)
        
    # 单样本热力图 (展示 Batch 中的第一个样本)
    visualize_heatmap_vertical(clean, delta, adv, sample_idx=0, save_path=f"vis_{args.config}_heatmap.png")