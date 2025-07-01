import os, sys
sys.path.append("./")
sys.path.append("../")

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.logger import logger

logger.is_debug(True)


if __name__ == "__main__":
    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi", 
        batch_size=8
    )

    texts, labels = next(iter(data_resource.train_loader))
    logger.debug(f"texts shape = {texts.shape}")
    logger.debug(f"单个文本样本: {texts[0]}")

    # 转为 sklean 支持的格式
    logger.debug(data_resource.X_train[0])
    logger.debug(data_resource.X_test[0])
    
    X_train, X_test, y_train, y_test = data_resource.to_sklearn_dataset()

    logger.debug(f"训练集: {len(X_train)}, {len(y_train)}")
    logger.debug(f"测试集: {len(X_test)}, {len(y_test)}")
    





    