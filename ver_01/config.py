#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for MTD vs Seeker ARL Framework (v21.0)
"""

import torch
import pathlib

class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # [v21] 버전업: 동적 인센티브 강화
    LEVEL = "L21_DynamicIncentives_v1"

    # 학습/검증 파라미터
    META_UPDATES = 500
    ROLLOUT_STEPS = 256
    VALIDATION_STEPS = 4096
    N_ENVS = 64

    # PPO
    LR = 3e-4; GAMMA = 0.99; GAE_LAMBDA = 0.95; K_EPOCHS = 10; EPS_CLIP = 0.2; MAX_GRAD_NORM = 0.5
    # [v20] ENTROPY_COEF 증가 (탐험 강화) - 유지
    ENTROPY_COEF = 0.03

    # 환경: IP/Port 공간
    NUM_IPS = 16
    SCENARIO = {"COMMON_PORTS": [14550, 14551, 5760, 5600, 7777, 8888, 9000]}
    NUM_ENDPOINTS = NUM_IPS * len(SCENARIO["COMMON_PORTS"])

    # Real-time 로깅
    TIME_STEP_SEC = 1.0 # 1스텝 = 1초로 가정
    TIME_BIN_SEC = 5    # 5초 단위로 집계

    # 동적 파라미터(Defender/Seeker)
    DYN_PARAMS = {
        "ip_cd":       {"base": 30.0, "min": 5.0,   "max": 60.0},
        "decoy_ratio": {"base": 0.05, "min": 0.0,   "max": 0.50},
        "bl_level":    {"base": 1.0,  "min": 0.0,   "max": 5.0},
    }
    SEEKER_PARAMS = {
        # (v19) 0.0 = Stealthy, 1.0 = Loud
        "attack_bias": {"base": 0.5,  "min": 0.0,   "max": 1.0}, 
        "scan_effort": {"base": 1.0,  "min": 0.5,   "max": 2.0},
    }

    # [v21] 다단계 공격 보상/비용/페널티 조정
    COST_WEIGHT = 0.25
    # 1차 방어 (Exploit)
    REW_MTD_BLOCK_EXPLOIT = 1.0
    # [v21] 디코이 보상 추가 증가
    REW_MTD_DECOY = 1.5 # 1.0에서 증가
    # [v21] 1차 침투 페널티 증가
    PENALTY_MTD_EXPLOIT = -3.0 # -2.0에서 증가
    # 2차 방어 (Breach)
    REW_MTD_BLOCK_BREACH = 2.0
    PENALTY_MTD_BREACH = -5.0 # 유지
    # [v21] 발각 페널티 증가
    PENALTY_MTD_KNOWLEDGE_LEAK = -0.5 # -0.1에서 증가

    COST_MTD_ACTION = 0.05
    # [v21] 셔플 비용 추가 감소
    COST_SHUFFLE = 0.05 # 0.10에서 감소
    COST_DECOY_RATIO = 0.15 # 유지
    COST_BL_LEVEL = 0.25 # 유지 (v20에서 증가됨)

    # [v19] Seeker 행동 파라미터 (유지)
    KNOWLEDGE_EXPLOIT_P = 0.85
    BLIND_EXPLOIT_P = 0.05
    EXPLOITED_BREACH_P = 0.90

    # [v19] 공격 유형 및 방어 효과 (v20 너프 유지)
    LOUD_EXPLOIT_P_BASE = 0.8
    STEALTHY_EXPLOIT_P_BASE = 0.5

    BLOCK_LOUD_K = 0.9
    BLOCK_LOUD_C = -0.5
    BLOCK_STEALTHY_K = 0.2
    BLOCK_STEALTHY_C = -1.5
    BLOCK_BREACH_K = 0.3
    BLOCK_BREACH_C = -1.0
    
    # 메타 액션 (유지)
    MTD_META_ACTIONS = {
        0: ("ip_cd", 1.2),  1: ("ip_cd", 0.8),
        2: ("decoy_ratio", 1.2), 3: ("decoy_ratio", 0.8),
        4: ("bl_level", 1.0), 5: ("bl_level", -1.0),
        6: ("none", 1.0),
    }
    SEEKER_META_ACTIONS = {
        0: ("attack_bias", 1.25), 1: ("attack_bias", 0.8),
        2: ("scan_effort", 1.2), 3: ("scan_effort", 0.8),
        4: ("none", 1.0),
    }
    MTD_META_ACTION_DIM = len(MTD_META_ACTIONS)
    SEEKER_META_ACTION_DIM = len(SEEKER_META_ACTIONS)

    # 정적 MTD 레벨 (유지)
    STATIC_MTD_LEVELS = {
        0: {"name": "L0 (Passive)",    "ip_cd": 60.0, "decoy_ratio": 0.00, "bl_level": 0.0},
        1: {"name": "L1 (Low)",        "ip_cd": 45.0, "decoy_ratio": 0.05, "bl_level": 1.0},
        2: {"name": "L2 (Medium)",     "ip_cd": 30.0, "decoy_ratio": 0.15, "bl_level": 2.0},
        3: {"name": "L3 (High)",       "ip_cd": 15.0, "decoy_ratio": 0.25, "bl_level": 3.0},
        4: {"name": "L4 (Very High)",  "ip_cd":  7.5, "decoy_ratio": 0.35, "bl_level": 4.0},
        5: {"name": "L5 (Max)",        "ip_cd":  5.0, "decoy_ratio": 0.50, "bl_level": 5.0},
    }
    # Seeker 행동 레벨 (유지)
    SEEKER_BEHAVIOR_LEVELS = {
        0: {"name": "L0 (Naive)",      "scan_effort": 0.5, "attack_bias": 0.5},
        1: {"name": "L1 (Scanner)",    "scan_effort": 2.0, "attack_bias": 0.8},
        2: {"name": "L2 (Stealthy)",   "scan_effort": 0.8, "attack_bias": 0.2},
        3: {"name": "L3 (ARL)",        "mode": "arl"},
    }

    # CTI 이벤트 입력 파일 (유지)
    CTI_EVENT_PATH = "ml/output/cti_event.json"

    # 내부 CTI 트리거 (유지)
    CTI_INTERNAL_TRIGGER = False
    CTI_INTERNAL_WINDOW_STEPS = 60
    CTI_INTERNAL_SUCCESS_COUNT = 5 # Breach 기준
    CTI_INTERNAL_RESPONSE = {
        "ip_cd_mult": 0.8, 
        "decoy_add": 0.10, 
        "bl_add": 1.0
    }

# config 인스턴스를 생성하여 다른 파일에서 import하여 사용
config = Config()