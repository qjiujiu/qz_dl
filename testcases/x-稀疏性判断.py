import os, sys
sys.path.append("./")
sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger

# 如果 rank ≈ 128：说明这个矩阵的列是线性无关的，信息丰富
# 如果 rank << 128：说明矩阵中有很多冗余信息，特征非常稀疏
# 如果 rank == 128：说明矩阵是满秩的

if __name__ == '__main__':
    datasets = ["malapi_cleanemb", "malapi_fgsmemb", "malapi_pgdemb"]
    for dataset in datasets:
        data_resource = TextDataSrc.load_dataset(
            dataset_name=dataset, 
            batch_size=8, 
        )


        total_rank = 0.0
        total_samples = 0

        with torch.no_grad():  # 不需要梯度，节省内存
            for x, y in data_resource.train_loader:
                batch_size = x.shape[0]
                
                for i in range(batch_size):
                    matrix = x[i]  # shape: [200, 128]
                    rank = torch.linalg.matrix_rank(matrix)
                    total_rank += rank.item()
                    total_samples += 1

        avg_rank = total_rank / total_samples
        print(f"数据集: {dataset} 平均秩（Across {total_samples} samples）: {avg_rank:.2f}")