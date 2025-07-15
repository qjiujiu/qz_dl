import os, sys
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger

from utils.models import (
    pick_model, 
    pick_embedding_encoder
)

# 数据集 rank 分析
# 如果 rank ≈ 128：说明这个矩阵的列是线性无关的，信息丰富
# 如果 rank << 128：说明矩阵中有很多冗余信息，特征非常稀疏
# 如果 rank == 128：说明矩阵是满秩的

# 数据集: malapi_cleanemb
# 平均秩（Across 5685 samples）: 20.17
# 类别分布:
#     类别 0:  (11.79%)
#     类别 1:  (14.11%)
#     类别 2:  (14.09%)
#     类别 3:  (13.90%)
#     类别 4:  (5.42%)
#     类别 5:  (12.49%)
#     类别 6:  (14.18%)
#     类别 7:  (14.04%)

def analyze_dataset():
    datasets = ["malapi_cleanemb", "malapi_fgsmemb", "malapi_pgdemb"]
    
    for dataset in datasets:
        data_resource = TextDataSrc.load_dataset(
            dataset_name=dataset, 
            batch_size=1024, 
        )

        total_rank = 0.0
        total_samples = 0
        
        # 新增：用于统计类别分布
        class_counts = {}
        with torch.no_grad():  # 不需要梯度，节省内存
            for x, y in data_resource.train_loader:
                batch_size = x.shape[0]
                total_samples += batch_size

                # 计算矩阵秩（原功能）
                for i in range(batch_size):
                    matrix = x[i]  # shape: [200, 128]
                    rank = torch.linalg.matrix_rank(matrix)
                    total_rank += rank.item()

                # 统计类别分布
                labels = y.tolist()
                for label in labels:
                    class_counts[label] = class_counts.get(label, 0) + 1
                 
        # 计算平均秩
        avg_rank = total_rank / total_samples

        # 排序类别编号
        sorted_classes = sorted(class_counts.keys())
        
        # 输出信息
        print(f"数据集: {dataset}")
        print(f"平均秩（Across {total_samples} samples）: {avg_rank:.2f}")
        print(f"类别分布:")
        for sc in sorted_classes:
            count = class_counts[sc]
            percent = count / total_samples * 100
            print(f"    类别 {sc}:  ({percent:.2f}%)")
        print("-" * 50)


# 无论 fgsm 或 pgd，训练之后几乎都是满秩的
def analyze_model(cfg):
    model = pick_model(cfg, cfg.checkpoint_path)
    logger.debug(f"模型已加载，权重路径:{cfg.checkpoint_path}，模型结构: {model}")
    
    # 假设模型.embedding.weight 形状 [vocab_size, embedding_dim]
    embedding_weight = model.embedding.weight.data 

    # 分析整个 embedding 矩阵的秩
    full_rank = torch.linalg.matrix_rank(embedding_weight).item()

    # 分析每个 token 向量的秩（即对每一行视为一个矩阵）其实是1，因为是单向量
    # 更有意义的是对 embedding 的局部子块做 rank 分析

    # 可选分析：随机采样若干行（子集）计算 rank 分布
    vocab_size = embedding_weight.shape[0]
    sample_size = min(200, vocab_size)  # 最多采样200个词
    indices = torch.randperm(vocab_size)[:sample_size]

    total_sub_rank = 0
    for i in range(0, sample_size, 10):  # 每10个向量为一个子矩阵
        sub_matrix = embedding_weight[indices[i:i+10]]  # shape: [10, embedding_dim]
        if sub_matrix.shape[0] > 1:
            rank = torch.linalg.matrix_rank(sub_matrix)
            total_sub_rank += rank.item()

    avg_local_rank = total_sub_rank / (sample_size // 10)

    print(f"模型: {cfg.model}")
    print(f"Embedding 总维度: {embedding_weight.shape}")
    print(f"整个 embedding 矩阵rank: {full_rank}")
    print(f"随机子集的平均 rank（每10个token一组）: {avg_local_rank:.2f}")




if __name__ == '__main__':
    analyze_dataset()