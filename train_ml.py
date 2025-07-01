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


logger.is_debug(True)


def train_and_evaluate(model, model_name, X_train_vec, y_train, X_test_vec, y_test):
    print(f"Training {model_name}...")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 60)


# python train_lstm.py --batch-size 8 --epochs 30 --lr 0.001 --dropout-prob 0.5  --embedding-dim 128 --hidden-dim 256 --output-dim 8  --max-len 200  --vocab-size 278
if __name__ == "__main__":
    cfg =  ArgsParser().create_nlp_config()
    data_resource = TextDataSrc.load_dataset(
        dataset_name="malapi", 
        batch_size=cfg.batch_size
    )

    X_train, X_test, y_train, y_test = data_resource.to_sklearn_dataset()
    
    SAMPLES = len(X_train)
    X_train, X_test, y_train, y_test = X_train[:SAMPLES], X_test[:SAMPLES], y_train[:SAMPLES], y_test[:SAMPLES]

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=128)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    

    print("TF-IDF Feature Shape:", X_train_tfidf.shape)

    # 定义要训练的模型
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100)),
        ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
        ("LightGBM", LGBMClassifier()),
        ("Linear SVM", LinearSVC())
    ]

    # 训练所有模型
    for name, model in models:
        train_and_evaluate(model, name, X_train_tfidf, y_train, X_test_tfidf, y_test)


