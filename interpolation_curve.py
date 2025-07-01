import torch
import matplotlib.pyplot as plt
from config.datasets.clean_emb_loader import load_clean_embedding_dataset
from config.datasets.adv_emb_loader import load_adversarial_dataset
from models.nlp.lstm_text_classifier import LSTMTextClassifier
from utils.get_config import load_config
from tqdm import tqdm


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


def interpolate_models_and_evaluate(config, model_path_a, model_path_b, test_loader, device, interpolate_keys=None):
    model_a = create_model(config).to(device)
    model_b = create_model(config).to(device)

    model_a.load_state_dict(torch.load(model_path_a, map_location=device), strict=False)
    model_b.load_state_dict(torch.load(model_path_b, map_location=device), strict=False)

    # w_values = [round(w, 1) for w in torch.arange(0.1, 1.0, 0.1).tolist()]
    w_values = [round(w, 2) for w in torch.arange(0.01, 1.0, 0.01).tolist()]
    accuracies = []

    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    if interpolate_keys is None:
        interpolate_keys = state_dict_a.keys()  # 默认插值所有参数

    for w in tqdm(w_values, desc="Interpolating and Evaluating"):
        mixed_model = create_model(config).to(device)
        mixed_state = {}

        for key in state_dict_a:
            if key in interpolate_keys:
                mixed_state[key] = w * state_dict_a[key] + (1 - w) * state_dict_b[key]
            else:
                mixed_state[key] = state_dict_a[key]  # 或者 state_dict_b[key]

        mixed_model.load_state_dict(mixed_state, strict=False)
        acc = evaluate(mixed_model, test_loader, device)
        accuracies.append(acc)

    return w_values, accuracies

def run_layerwise_interpolation_two_datasets(
    config_path,
    model_path_a,
    model_path_b,
    test_loaders: dict,  # {"clean": loader1, "adv": loader2}
    save_path: str
):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 所有参数名
    all_keys = [
        "embedding.weight",
        "lstm.weight_ih_l0", "lstm.weight_hh_l0",
        "lstm.bias_ih_l0", "lstm.bias_hh_l0",
        "fc.weight", "fc.bias"
    ]

    groups = {
        "embedding": ["embedding.weight"],
        "lstm": ["lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0", "lstm.bias_hh_l0"],
        "fc": ["fc.weight", "fc.bias"]
    }

    combinations = [
        ("no_embedding", groups["lstm"] + groups["fc"]),
        ("no_lstm", groups["embedding"] + groups["fc"]),
        ("no_fc", groups["embedding"] + groups["lstm"]),
        ("all", all_keys)
    ]

    plt.figure(figsize=(10, 6))
    all_w_values = None

    # 颜色按组合分配
    colors = {
        "no_embedding": "#6EC1E4",  # 清新蓝
        "no_lstm": "#8CD17D",       # 薄荷绿
        "no_fc": "#F6C85F",         # 柔橙黄
        "all": "#D4A6C8"            # 淡紫色
    }

    for combo_name, interpolate_keys in combinations:
        color = colors[combo_name]
        show_label = True  # 只在第一条线显示 legend

        for ds_name, loader in test_loaders.items():
            w_values, accs = interpolate_models_and_evaluate(
                config, model_path_a, model_path_b, loader, device, interpolate_keys
            )

            if all_w_values is None:
                all_w_values = w_values

            # 主线条
            plt.plot(
                w_values,
                accs,
                color=color,
                marker='.',
                markersize=3,
                linewidth=1,
                alpha=0.85,
                label=combo_name if show_label else None
            )

            plt.annotate(
                ds_name,
                xy=(w_values[-1], accs[-1]),
                xytext=(5, 0),  # x方向偏移 5px
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center"
            )

            # 在每条线尾部添加文字标注（clean / adv）
            plt.text(
                w_values[-1] + 0.01, accs[-1],  # 稍微偏右一点
                ds_name,
                fontsize=8,
                color=color,
                verticalalignment='center'
            )

            show_label = False  # 仅第一条线使用 label（避免图例重复）

    # 图设置
    plt.xlabel("Weight w in wa + (1-w)b")
    plt.ylabel("Accuracy")
    plt.title("Interpolation Across Layer Combinations (Clean vs Adv)")

    if len(all_w_values) <= 10:
        plt.xticks(ticks=all_w_values)
    else:
        plt.xticks(ticks=all_w_values[::max(1, len(all_w_values)//10)])

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Weight Combination", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



if __name__ == "__main__":

    config_path = "config/clean_adv_config.yaml"
    model_path_a = "checkpoints/lstm.pth"
    model_path_b = "checkpoints/lstm_adv_fgsm.pth"  # or lstm_adv_pgd.pth
    save_path = "figure/layerwise_interp.png"

    config = load_config(config_path)

    _, clean_loader = load_clean_embedding_dataset(
        cache_path="data/malapi2019/emb-feature/clean-exam/clean_examples.pt",
        batch_size=config["batch_size"]
    )

    _, adv_loader = load_adversarial_dataset(
        emb_path="data/malapi2019/emb-feature/advexam-fgsm/fgsm.pt",  # or pgd.pt
        batch_size=config["batch_size"]
    )

    test_loaders = {
        "clean": clean_loader,
        "adv": adv_loader
    }

    run_layerwise_interpolation_two_datasets(
        config_path,
        model_path_a,
        model_path_b,
        test_loaders,
        save_path
    )