# File: ver_02/utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # (필요시) CuDNN 설정
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def Array2Tensor(array, device):
    """Numpy 배열을 PyTorch 텐서로 변환"""
    if not isinstance(array, np.ndarray):
        array = np.array(array)
        
    if array.dtype == np.int64:
        return torch.LongTensor(array).to(device)
    else:
        return torch.FloatTensor(array).to(device)

if __name__ == '__main__':
    set_seed(42)
    test_arr_float = [0.1, 0.2]
    test_arr_int = [1, 2]
    
    tensor_float = Array2Tensor(test_arr_float, 'cpu')
    tensor_int = Array2Tensor(test_arr_int, 'cpu')
    
    print(f"Float Array -> Tensor: {tensor_float} (Type: {tensor_float.dtype})")
    print(f"Int Array -> Tensor: {tensor_int} (Type: {tensor_int.dtype})")
    print("Seed set to 42.")