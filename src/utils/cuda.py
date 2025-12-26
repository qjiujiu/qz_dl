import torch
import logging

def select_gpu() -> str:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # 获取各 GPU 空闲显存（单位：MiB）
    import subprocess
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
        capture_output=True, 
        text=True
    )
    
    free_memory = [int(x) for x in result.stdout.strip().split('\n')]
    best_gpu = int(torch.argmax(torch.tensor(free_memory)).item())
    
    # 自动获取剩余空间最多的 GPU
    logging.info(f"Auto-selected GPU {best_gpu} with {free_memory[best_gpu]} MiB free")
    return f"cuda:{best_gpu}"