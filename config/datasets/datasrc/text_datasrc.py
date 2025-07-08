from config.datasets.dataset_instance import mal_api
from config.datasets.data_resource import DataResource

def load_malapi(dataset_name, batch_size = 8):        
    # 数据加载器模块直接返回两个 loader
    try:
        X_train, X_test, y_train, y_test, vocab  = mal_api.load(
            text_path="data/malapi2019/all_analysis_data.txt", 
            labels_path="data/malapi2019/labels.txt"
        )

        # 创建训练和测试数据集
        train_dataset = mal_api.MalAPITextDataset(texts=X_train, labels=y_train, vocab=vocab)
        test_dataset = mal_api.MalAPITextDataset(texts=X_test, labels=y_test, vocab=vocab)
                    
        return DataResource(
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            batch_size=batch_size,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vocab=vocab
        )
        
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found in 'datasrc' modules!")

def load_malapi_fgsmemb(dataset_name, batch_size = 8):
    try:
        X_train, X_test, y_train, y_test  = mal_api.load_fgsmemb()
        train_dataset = mal_api.MalAPIEmbedDataset(embeddings=X_train, labels=y_train)
        test_dataset = mal_api.MalAPIEmbedDataset(embeddings=X_test, labels=y_test)
        return DataResource(
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            batch_size=batch_size,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
    
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found in 'datasrc' modules!")
    
    
def load_malapi_pgdemb(dataset_name, batch_size = 8):
    try:
        X_train, X_test, y_train, y_test  = mal_api.load_pgdemb()
        train_dataset = mal_api.MalAPIEmbedDataset(embeddings=X_train, labels=y_train)
        test_dataset = mal_api.MalAPIEmbedDataset(embeddings=X_test, labels=y_test)
        return DataResource(
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            batch_size=batch_size,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
        )
    
    except ModuleNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found in 'datasrc' modules!")

class TextDataSrc:
    @staticmethod
    def load_dataset(dataset_name, batch_size = 8, encoder = None):
        """ 根据数据集名称加载对应的数据集模块，并返回训练集、测试集和词汇表。
            返回一个DataResource 模块，包含两个 loader
        """
        if dataset_name == "malapi":
            return load_malapi(dataset_name, batch_size = 8)
        elif dataset_name == "malapi_fgsmemb":
            return load_malapi_fgsmemb(dataset_name, batch_size = 8)
        elif dataset_name == "malapi_pgdemb":
            return load_malapi_pgdemb(dataset_name, batch_size = 8)
    
