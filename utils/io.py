import os
import torch
import pickle

from config.logger import logger

def read_pickle(fname):
    """读取pickle文件并返回数据"""
    with open(fname, 'rb') as f:
        return pickle.load(f)

def write_pickle(fname, data):
    """将数据写入pickle文件"""
    with open(fname, 'wb') as f:
        pickle.dump(data, f)





def save_model_weights(model, fpath):
    """ 保存 PyTorch 模型的权重到指定路径。
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        
        # 保存模型状态字典
        torch.save(model.state_dict(), fpath)
        logger.success(f"模型权重已成功保存至: {fpath}")
    except Exception as e:
        logger.error(f"保存模型权重时出错: {e}")


def load_model_weights(model, load_path, device='cpu'):
    """ 通过指定路径加载 PyTorch 模型的权重。
    """
    try:
        # 加载模型权重
        state_dict = torch.load(load_path, map_location=device)
        
        # 加载权重到模型中
        model.load_state_dict(state_dict)
        logger.success(f"模型权重已成功加载自: {load_path}")
        return model
    except Exception as e:
        # 出错时返回原始未加载权重的模型
        logger.error(f"加载模型权重时出错: {e}")
        return model  
