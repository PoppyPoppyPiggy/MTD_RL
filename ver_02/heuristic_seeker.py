# File: ver_02/heuristic_seeker.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic Seeker (v23 - 배포 환경 호환)

[수정]
이 Seeker는 v23 환경(environment.py)과 호환됩니다.
MTD 파라미터(ip_cd, decoy_ratio) 대신, 
'active_mtd_id' (0-6)를 직접 관찰하여 행동을 결정합니다.

- active_mtd_id 0~5: Real Target (공격 대상)
- active_mtd_id 6: Decoy (공격 회피 대상)
"""

import random

class HeuristicSeeker:
    def __init__(self, args):
        self.level = args.seeker_level
        self.knowledge_threshold = 0.5 # Exploit 시도에 필요한 최소 지식
        self.reset()
        
    def reset(self):
        self.last_scan_result = False # 스캔 성공 여부

    def select_action(self, attacker_knowledge, active_mtd_id):
        """
        [수정] v23 환경에 맞는 행동 결정
        Args:
            attacker_knowledge (float): 0.0 ~ 1.0
            active_mtd_id (int): 0~6 (0-5=Real, 6=Decoy)
        """
        
        # 현재 MTD가 Decoy인지 Real인지 확인
        is_decoy_active = (active_mtd_id == 6)
        
        # L0: No-Op
        if self.level == 'L0':
            return 'pass'
        
        # L1: Always Scan
        if self.level == 'L1':
            return 'scan'
            
        # L2: Naive Scan & Exploit
        # 지식이 쌓이면 Decoy 여부와 상관없이 Exploit 시도
        if self.level == 'L2':
            if attacker_knowledge < self.knowledge_threshold:
                return 'scan'
            else:
                return 'exploit'
        
        # L3: State-Aware Scan & Exploit (Decoy 인지)
        # Decoy가 활성화(ID 6)되면 Exploit을 멈추고, Real이 활성화되면 공격
        if self.level == 'L3':
            if is_decoy_active:
                # Decoy가 켜져있으므로, Exploit은 무의미함.
                # 지식을 쌓기 위해 'scan'을 하거나, 비용을 아끼기 위해 'pass'
                return 'scan' 
            else:
                # Real Target이 켜져있음
                if attacker_knowledge < self.knowledge_threshold:
                    return 'scan'
                else:
                    return 'exploit' # L2와 동일하게 공격

        # L4: Adaptive (L3와 동일, L4만의 고유 로직 추가 가능)
        # v23 환경에서는 L3와 L4의 시뮬레이션 로직이 동일함
        # (v21의 bl_level 파라미터가 v23에는 없기 때문)
        if self.level == 'L4':
            if is_decoy_active:
                return 'scan' # Decoy 상태에서는 스캔하여 상태 변경 감지
            else:
                # Real Target
                if attacker_knowledge < self.knowledge_threshold:
                    return 'scan'
                else:
                    return 'exploit' # 즉시 공격

        return 'pass'