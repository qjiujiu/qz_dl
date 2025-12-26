
from dataclasses import is_dataclass, asdict
from pydantic import BaseModel
from pathlib import Path
from typing import Any, List, Dict, Tuple

import json
import pickle
import logging

def ensure_dirs(*dirs: str):
    for d in dirs:
        if d is not None:
            Path(d).mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Dict:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if is_dataclass(obj):
        return to_jsonable(asdict(obj)) 
    elif isinstance(obj, list):
        return [to_jsonable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    else:
        return obj


def dump_to_json(data: Dict, path: Path):
    """通用方法：能把数据保存到指定路径的 JSON 文件中
        1. 使用 `to_jsonable` 转换数据（如果需要）
        2. 使用 `json.dump` 保存为 JSON 文件
        参数：
        - data: 要保存的数据（通常是字典或列表）
        - path: 保存 JSON 文件的路径
    """
    # 转换数据为可序列化的格式
    jsonable_data = to_jsonable(data)
    
    # 存为 JSON 文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(jsonable_data, f, indent=4, ensure_ascii=False)
    
    logging.info(f"Data saved to {path}")


def read_pickle(*fnames: str) -> List:
    """批量读取多个pickle文件并返回数据"""
    data = []
    for fname in fnames:
        with open(fname, 'rb') as f:
            data.append(pickle.load(f))
    return data

# 批量写入
def write_pickle(*fname_data_pairs):
    """批量将数据写入pickle文件"""
    for fname, data in fname_data_pairs:
        with open(fname, 'wb') as f:
            pickle.dump(data, f)
        