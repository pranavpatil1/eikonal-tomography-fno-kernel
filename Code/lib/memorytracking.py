import psutil
import torch
import time
import pickle
import matplotlib.pyplot as plt

def get_cpu_usage_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def get_gpu_usage_mb():
    return torch.cuda.memory_allocated() / (1024 * 1024)

def save_memory_usage(arr, event, gpu=False, savefile=None):
    arr.append({
        "time": time.time(),
        "event": event,
        "cpu": get_cpu_usage_mb(),
        "gpu": get_gpu_usage_mb(),
    })
    if savefile:
        with open(savefile, 'wb') as file: 
            pickle.dump(arr, file)

def save_visualization(infile, outfile):
    with open(infile, 'rb') as f:
        memorydata = pickle.loads(f.read())
    
    