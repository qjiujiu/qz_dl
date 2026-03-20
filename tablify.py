import csv
import glob
import os
import statistics
from typing import List, Dict, Any
from pathlib import Path

# 假设你的项目结构中已有这些模块
from src.utils.collections import KVTree
from src.utils.dumps import load_from_json

# ================= 配置区域 =================
TARGET_DIR = './outputs/trace/maldynamic/2025-12-31'  # 目标目录
OUTPUT_CSV = 'experiment_results.csv'

# 排序规则：列表中的 key 对应 CSV 的列名（或 row_data 的 key）
# 优先按第一个 key 排序，如果相同则按第二个，以此类推。
SORT_KEYS = ["dataset_name", "network", "attention", "task_id"]
# ===========================================

def calculate_mean_std(values: List[float]) -> str:
    """计算均值和标准差，返回格式字符串 'Mean ± Std'"""
    if not values:
        return "N/A"
    
    # 过滤掉 None 值，防止数据缺失导致的报错
    valid_values = [v for v in values if v is not None]
    
    if not valid_values:
        return "N/A"

    if len(valid_values) == 1:
        return f"{valid_values[0]:.2f} ± 0.00"
    
    mean_val = statistics.mean(valid_values)
    stdev_val = statistics.stdev(valid_values)
    return f"{mean_val:.2f} ± {stdev_val:.2f}"

def extract_metrics_from_file(file_path: str) -> Dict[str, Any]:
    """读取单个 JSON 文件并提取所需字段"""
    try:
        raw_data = load_from_json(file_path)
        tree = KVTree(raw_data)
        
        # 基础信息提取
        task_id = tree.get("task_id", "Unknown")
        dataset_name = tree.get("data_config.dataset_name", "Unknown")
        net_name = tree.get("network_config.name", "N/A")
        atten_type = tree.get("network_config.plugin_type", "N/A")

        # 核心指标处理 (val_metrics)
        val_metrics = tree.get("val_metrics", [])
        
        # 准备容器存储最后3次的具体数值
        metrics_storage = {
            "acc": [],        # micro.p
            "macro_p": [],    # macro.p
            "macro_r": [],    # macro.r
            "macro_f1": [],   # macro.f1
            "samples_f1": []  # samples_f1 (新增统计)
        }

        # 确保 val_metrics 是列表且不为空
        if isinstance(val_metrics, list) and len(val_metrics) > 0:
            # 取最后 3 个元素
            last_n_metrics = val_metrics[-3:]
            
            for metric_item in last_n_metrics:
                # 使用 KVTree 包装 item 以支持路径查找
                m_tree = KVTree(metric_item)
                
                # 提取各项数值
                metrics_storage["acc"].append(m_tree.get("micro.p", 0.0))
                metrics_storage["macro_p"].append(m_tree.get("micro.p", 0.0))
                metrics_storage["macro_r"].append(m_tree.get("micro.r", 0.0))
                metrics_storage["macro_f1"].append(m_tree.get("micro.f1", 0.0))
                
                # 提取 samples_f1 (假设它在每个 epoch 的 metric 对象里)
                # 如果 json 里层级是 "samples": {"f1": 99}, 则可以用 "samples.f1"
                metrics_storage["samples_f1"].append(m_tree.get("samples_f1", 0.0))

        # 3. 计算统计值 (Mean ± Std)
        row_data = {
            "task_id": task_id,
            "dataset_name": dataset_name,
            "network": net_name,
            "attention": atten_type,
            
            # 统计数据
            "acc": calculate_mean_std(metrics_storage["acc"]),
            "macro_p": calculate_mean_std(metrics_storage["macro_p"]),
            "macro_r": calculate_mean_std(metrics_storage["macro_r"]),
            "macro_f1": calculate_mean_std(metrics_storage["macro_f1"]),
            "samples_f1": calculate_mean_std(metrics_storage["samples_f1"]) # 新增
        }
        
        return row_data

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def tablify():
    # 递归查找所有 json 文件
    search_path = Path(TARGET_DIR)
    json_files = sorted(search_path.rglob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {TARGET_DIR}...")

    results = []
    
    # 提取数据
    for f in json_files:
        row = extract_metrics_from_file(str(f))
        if row:
            results.append(row)

    if not results:
        print("No valid data extracted.")
        return

    # 执行排序 (新增功能)
    if SORT_KEYS:
        print(f"Sorting results by: {SORT_KEYS}")
        try:
            # 使用 lambda 构建排序元组。
            # str() 确保 None 或数字都能被比较，
            # 如果需要按数字大小排序而不是字典序(例如 10 > 2)，需要在这里做类型转换
            results.sort(key=lambda x: tuple(str(x.get(k, "")) for k in SORT_KEYS))
        except Exception as e:
            print(f"Sorting failed: {e}. Check if SORT_KEYS matches data keys.")

    # 定义表头, 写入 CSV
    headers = [
        "task_id", "dataset_name", "network", "attention", 
        "acc", "macro_p", "macro_r", "macro_f1", "samples_f1"
    ]
    
    # 确保 key_map 与 headers 顺序对应
    key_map = [
        "task_id", "dataset_name", "network", "attention",
        "acc", "macro_p", "macro_r", "macro_f1", "samples_f1"
    ]

    try:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for res in results:
                writer.writerow([res.get(k) for k in key_map])
        
        print(f"Successfully generated CSV at: {os.path.abspath(OUTPUT_CSV)}")
        print(f"Total rows: {len(results)}")
        
    except IOError as e:
        print(f"Error writing CSV file: {e}")


if __name__ == '__main__':
    tablify()