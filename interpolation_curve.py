import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

from models.nlp.lstm_text_classifier import LSTMTextClassifier
from datasets.clean_emb_loader import load_clean_embedding_dataset
from tqdm import tqdm
from datasets.adv_emb_loader import load_adversarial_dataset

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config):
    return LSTMTextClassifier(
        vocab_size=278,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        max_len=config['max_len']
    )


def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def interpolate_models_and_evaluate(config, model_path_a, model_path_b, test_loader, device):
    model_a = create_model(config).to(device)
    model_b = create_model(config).to(device)

    model_a.load_state_dict(torch.load(model_path_a, map_location=device), strict=False)
    model_b.load_state_dict(torch.load(model_path_b, map_location=device), strict=False)

    # w_values = [round(w, 1) for w in torch.arange(0.1, 1.0, 0.1).tolist()] # 步长 0.1
    w_values = [round(w, 2) for w in torch.arange(0.01, 1.0, 0.01).tolist()] # 步长 0.01
    accuracies = []
    
    for w in tqdm(w_values, desc="Interpolating and Evaluating"):  
        mixed_model = create_model(config).to(device)
        mixed_state = {
            key: w * model_a.state_dict()[key] + (1 - w) * model_b.state_dict()[key]
            for key in model_a.state_dict()
        }
        mixed_model.load_state_dict(mixed_state, strict=False)
        acc = evaluate(mixed_model, test_loader, device)
        accuracies.append(acc)

    return w_values, accuracies


def plot_interpolation_curve(w_values_1, accuracies_1, w_values_2, accuracies_2, save_path, ylabel, title):
    plt.figure(figsize=(10, 6))

    # 绘制第一条线 (model_a + model_b 插值)
    plt.plot(w_values_1, accuracies_1, marker='.', markersize=3, linewidth=1, label="Clean Test Data")

    # 绘制第二条线 (可能是另一模型的插值)
    plt.plot(w_values_2, accuracies_2, marker='x', markersize=4, linewidth=1, label="Adv Test Data")

    # 添加标签、标题、网格等
    plt.xlabel("Weight w in wa + (1-w)b")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ticks=w_values_1[::10])  # 每隔0.1显示一次
    plt.grid(True, linestyle='--', alpha=0.6)

    # 添加图例
    plt.legend()

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_interpolation_experiment(config_path, model_path_a, model_path_b, test_data_path1, test_data_path2, save_path, ylabel, title):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader1 = load_clean_embedding_dataset(
        cache_path=test_data_path1,
        batch_size=config['batch_size']
    )

    _, test_loader2 = load_adversarial_dataset(
        emb_path=test_data_path2,
        batch_size=config['batch_size']
    )

    # 计算第一条线 (model_a 和 model_b 之间的插值)
    w_values_1, accuracies_1 = interpolate_models_and_evaluate(
        config, model_path_a, model_path_b, test_loader1, device
    )

    # 计算第二条线 (model_c 和 model_d 之间的插值)
    w_values_2, accuracies_2 = interpolate_models_and_evaluate(
        config, model_path_a, model_path_b, test_loader2, device
    )

    # 绘制两条线
    plot_interpolation_curve(w_values_1, accuracies_1, w_values_2, accuracies_2, save_path, ylabel, title)





if __name__ == "__main__":

    # # 干净模型+fgsm对抗模型
    # model_path_a = "checkpoints/lstm.pth"
    # model_path_b = "checkpoints/lstm_adv_fgsm.pth"  
    # # 干净embedding特征作为输入
    # test_data_path1 = "data/malapi2019/emb-feature/clean-exam/clean_examples.pt"
    # # fgsm对抗样本embedding特征作为输入
    # test_data_path2 = "data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt"

    # config_path = "config/clean_adv_config.yaml"                               # 用在干净模型+fgsm对抗模型的参数文件
    # save_path = "figure/comparison_clean_fgsm-trained_model.png"
    # ylabel = "Accuracy"
    # title = "Comparison between Clean and FGSM-trained Models"

    # run_interpolation_experiment(config_path, model_path_a, model_path_b, test_data_path1, test_data_path2, save_path, ylabel, title)


    # 干净模型+pgd对抗模型
    model_path_a = "checkpoints/lstm.pth"
    model_path_b = "checkpoints/lstm_adv_pgd.pth"  
    # 干净embedding特征作为输入
    test_data_path1 = "data/malapi2019/emb-feature/clean-exam/clean_examples.pt"
    # pgd对抗样本embedding特征作为输入
    test_data_path2 = "data/malapi2019/emb-feature/advexam-pgd/pgd.pt"

    config_path = "config/clean_adv_config.yaml"                            # 用在干净模型+pgd对抗模型的参数文件
    save_path = "figure/comparison_clean_pgd-trained_model.png"
    ylabel = "Accuracy"
    title = "Comparison between Clean and PGD-trained Models"

    run_interpolation_experiment(config_path, model_path_a, model_path_b, test_data_path1, test_data_path2, save_path, ylabel, title)