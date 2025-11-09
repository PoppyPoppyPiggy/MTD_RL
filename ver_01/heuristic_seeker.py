#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic Seeker Agent (v22)
- Testbed-Compatible
- Config_v21의 5-Level Seeker (L0~L4)를 구현
- MTD 파라미터를 인지하고 행동을 결정
"""

import random
import numpy as np

class HeuristicSeeker:
    
    # Config_v21의 5-Level Seeker (L0~L4) 정의
    SEEKER_BEHAVIOR_LEVELS = {
        # L0: 매우 소극적, 스캔/공격 거의 안함
        "L0": {"name": "L0 (Passive)", "scan_effort": 0.2, "attack_bias": 0.1},
        # L1: 소극적, 스캔 위주
        "L1": {"name": "L1 (Scanner)", "scan_effort": 0.5, "attack_bias": 0.3},
        # L2: 균형
        "L2": {"name": "L2 (Moderate)", "scan_effort": 1.0, "attack_bias": 0.5},
        # L3: 공격적, 스캔보다 공격 선호
        "L3": {"name": "L3 (Aggressive)", "scan_effort": 1.5, "attack_bias": 0.8},
        # L4: 매우 공격적, 지식 획득 시 즉시 공격
        "L4": {"name": "L4 (Exploiter)", "scan_effort": 1.0, "attack_bias": 1.0},
    }

    def __init__(self, args):
        self.args = args
        level_key = args.seeker_level
        
        if level_key not in self.SEEKER_BEHAVIOR_LEVELS:
            print(f"Warning: Seeker level '{level_key}' not found. Defaulting to L0.")
            level_key = 'L0'
            
        level_config = self.SEEKER_BEHAVIOR_LEVELS[level_key]
        
        self.scan_effort = level_config["scan_effort"] # 스캔 시도 빈도 (높을수록 자주 스캔)
        self.attack_bias = level_config["attack_bias"] # 공격 성향 (0=스캔만, 1=공격만)

        print(f"[Seeker] Initialized as {level_key} ({level_config['name']}): ScanEffort={self.scan_effort}, AttackBias={self.attack_bias}")
        
    def reset(self):
        pass

    def select_action(self, knowledge, mtd_params):
        """
        현재 지식과 MTD 상태를 기반으로 'scan', 'exploit', 'pass' 중 하나를 결정
        """
        
        # 1. 스캔할 것인가? (scan_effort 기반)
        if random.random() < self.scan_effort * (1.0 - knowledge): # 지식이 1.0이면 스캔 안함
            # 1a. 스캔 vs 공격 결정 (attack_bias 기반)
            if random.random() < self.attack_bias:
                return 'exploit' # 스캔 대신 공격 시도
            else:
                return 'scan'    # 스캔 시도
                
        # 2. 공격할 것인가? (지식이 있고, attack_bias가 높을수록)
        if knowledge > 0.1 and random.random() < self.attack_bias * knowledge:
            return 'exploit'
            
        # 3. 아무것도 안 함
        return 'pass'