#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for MTD vs Seeker ARL Framework (v21)
- [FIX] make_outdir가 'args' 인자를 받도록 수정
- [ADD] args를 json 파일로 저장하는 기능 추가
- [MERGE] Array2Tensor 함수 추가 (train_mtd_only.py 호환성)
"""

import random
import numpy as np
import torch
import pathlib
import datetime as dt
import argparse  # [FIX] argparse import 추가
import json      # [FIX] json import 추가

def set_seed(seed: int):
    """RNG 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_outdir(level: str, args: argparse.Namespace) -> pathlib.Path:
    """
    [FIX] 'args' 인자를 받도록 시그니처 수정
    결과 디렉토리를 생성하고, 실행 인자(args)를 JSON으로 저장합니다.
    """
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = pathlib.Path("results") / f"{ts}_{level}"
    
    # 모든 하위 디렉토리 생성
    (outdir / "figs").mkdir(parents=True, exist_ok=True)
    (outdir / "figs-rt").mkdir(parents=True, exist_ok=True) # figs-rt도 생성 확인
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    
    # [ADD] 실행 인자(args)를 JSON 파일로 저장
    try:
        # argparse.Namespace를 딕셔너리로 변환
        args_dict = vars(args)
        # pathlib.Path 객체 등 JSON 직렬화가 불가능한 객체를 문자열로 변환
        safe_args_dict = {k: str(v) if isinstance(v, (pathlib.Path)) else v for k, v in args_dict.items()}
        
        with open(outdir / "args_summary.json", "w", encoding="utf-8") as f:
            json.dump(safe_args_dict, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Could not save args_summary.json: {e}")

    return outdir

# --- [MERGE] train_mtd_only.py가 요구하는 함수 ---
def Array2Tensor(array, device):
    """NumPy 배열을 지정된 device의 FloatTensor로 변환합니다."""
    # .astype(np.float32)를 추가하여 PPO 모델이 기대하는 float32 타입으로 명시적으로 변환
    return torch.tensor(array.astype(np.float32), device=device)