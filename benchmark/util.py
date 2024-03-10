import torch
import time
import numpy as np


def cuda_time(function) -> tuple[float, any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return_value = function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000, return_value


def cpu_time(function) -> tuple[float, any]:
    before = time.time()
    return_value = function()
    return time.time() - before, return_value
