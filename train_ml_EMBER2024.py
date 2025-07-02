# train_lstm.py
from sklearn.feature_extraction.text import (
    CountVectorizer, 
    TfidfVectorizer, 
    HashingVectorizer
)

from config.datasets.datasrc.text_datasrc import TextDataSrc
from config.params_parser.parser import ArgsParser
from config.logger import logger


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import thrember

logger.is_debug(True)


def train_and_evaluate(model, model_name, X_train_vec, y_train, X_test_vec, y_test):
    print(f"Training {model_name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    
    # 使用 format 参数控制输出精度
    report = classification_report(
        y_test, 
        y_pred, 
        output_dict=False, 
        digits=4  # 保留小数点后四位
    )
    print(report)
    print("-" * 60)

# python train_ml.py --max-feature 128
if __name__ == "__main__":
    logger.debug("训练开始...")

    X_train, y_train = thrember.read_vectorized_features('./data/EMBER2024/APK_all/', subset="train")
    logger.debug(f"训练集尺寸: {X_train.shape}, {y_train.shape}")

    X_test, y_test = thrember.read_vectorized_features('./data/EMBER2024/APK_all/', subset="test")
    logger.debug(f"测试集尺寸: {X_test.shape}, {y_test.shape}")


    # 定义要训练的模型
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ("LightGBM", LGBMClassifier()),
        ("Linear SVM", LinearSVC())
    ]

    # 训练所有模型
    for name, model in models:
        train_and_evaluate(model, name, X_train, y_train,X_test, y_test)


