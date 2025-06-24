import os
# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm 

# 加载 all-MiniLM-L6-v2 模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 标签映射
label_map = {
    'Spyware': 0,
    'Downloader': 1,
    'Trojan': 2,
    'Worms': 3,
    'Adware': 4,
    'Dropper': 5,
    'Virus': 6,
    'Backdoor': 7
}

# 读取样本和标签
def load_data(samples_file, labels_file):
    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = f.readlines()
    
    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    # 去除每行的换行符
    samples = [line.strip() for line in samples]
    labels = [line.strip() for line in labels]
    
    return samples, labels

# 获取样本和标签的 Embedding，并将标签转换为整数
def get_embeddings(samples, labels, model, label_map):
    embeddings = []
    mapped_labels = []
    
    # 使用 tqdm 显示进度
    for sample, label in tqdm(zip(samples, labels), total=len(samples), desc="Processing"):
        # 获取每个样本的嵌入
        embedding = model.encode(sample, convert_to_tensor=True)
        embeddings.append(embedding)
        
        # 将标签映射到整数
        mapped_labels.append(label_map[label])
    
    # 将嵌入转化为 Tensor
    embeddings = torch.stack(embeddings)
    
    return embeddings, mapped_labels

# 保存 Embedding 和标签到文件
def save_embeddings_and_labels(embeddings, labels, output_path):
    # 打包嵌入和标签作为元组
    cached_emb = embeddings
    cached_labels = torch.tensor(labels, dtype=torch.long)
    
    # 保存到指定路径
    torch.save((cached_emb, cached_labels), output_path)
    print(f"Embeddings and labels saved to {output_path}")

# 主函数
def main():
    # 数据文件路径
    samples_file = 'data/malapi2019/all_analysis_data.txt'
    labels_file = 'data/malapi2019/labels.txt'
    
    # 输出文件路径
    output_path = 'data/malapi2019/emb-MiniLM-L6/clean_examples.pt'
    
    # 加载数据
    samples, labels = load_data(samples_file, labels_file)
    
    # 获取文本的 Embedding 和对应的标签
    embeddings, mapped_labels = get_embeddings(samples, labels, model, label_map)
    
    # 保存 Embedding 和标签
    save_embeddings_and_labels(embeddings, mapped_labels, output_path)

if __name__ == "__main__":
    main()
