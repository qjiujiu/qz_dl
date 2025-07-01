import os
import torch
from config.datasets.dataset_instance import mal_api
from torch.utils.data import (
    TensorDataset, 
    DataLoader, 
    random_split
)

from config.datasets.data_resource import DataResource


class TextDataSrc:
    @staticmethod
    def load_dataset(dataset_name, batch_size = 8):
        """ 根据数据集名称加载对应的数据集模块，并返回训练集、测试集和词汇表。
            返回一个DataResource 模块，包含两个 loader
        """
        # 数据加载器模块直接返回两个 loader
        try:
            train_dataset, test_dataset, vocab = mal_api.load(
                text_file="data/malapi2019/all_analysis_data.txt", 
                labels_file="data/malapi2019/labels.txt"
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            return DataResource(train_loader=train_loader, test_loader=test_loader, vocab=vocab)
        
        except ModuleNotFoundError:
            raise ValueError(f"Dataset '{dataset_name}' not found in 'datasrc' modules!")

