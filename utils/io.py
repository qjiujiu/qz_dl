import pickle

def read_pickle(fname):
    """读取pickle文件并返回数据"""
    with open(fname, 'rb') as f:
        return pickle.load(f)

def write_pickle(fname, data):
    """将数据写入pickle文件"""
    with open(fname, 'wb') as f:
        pickle.dump(data, f)