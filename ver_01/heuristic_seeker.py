import numpy as np
import random

# (Config_v21) Seeker 행동 레벨 (L0: Naive, L1: Scanner, L2: Stealthy)
# --- [수정] Config_v21의 SEEKER_PARAMS 범위를 기반으로 5단계 강도 정의 ---
# 'scan_effort' (min: 0.5, max: 2.0)
# 'attack_bias' (min: 0.0, max: 1.0)
SEEKER_BEHAVIOR_LEVELS = {
    # L0 (매우 은밀): 스캔 최소, 공격 성향 최소
    "L0": {"name": "L0 (Stealthy)",   "scan_effort": 0.8, "attack_bias": 0.2}, 
    # L1 (소극적): 스캔 낮음, 공격 성향 낮음
    "L1": {"name": "L1 (Low)",        "scan_effort": 0.5, "attack_bias": 0.3},
    # L2 (중간): 기본값
    "L2": {"name": "L2 (Moderate)",   "scan_effort": 1.0, "attack_bias": 0.5},
    # L3 (공격적): 스캔 높음, 공격 성향 높음
    "L3": {"name": "L3 (Aggressive)", "scan_effort": 1.5, "attack_bias": 0.8},
    # L4 (스캐너): 스캔 최대, 공격 성향 최대
    "L4": {"name": "L4 (Scanner)",    "scan_effort": 2.0, "attack_bias": 1.0}
}

class HeuristicSeeker:
    def __init__(self, args):
        # --- [수정] Seeker 레벨 기본값을 L2(중간)로 변경 ---
        level_config = SEEKER_BEHAVIOR_LEVELS.get(args.seeker_level, SEEKER_BEHAVIOR_LEVELS["L2"])
        
        # L0: Naive, L1: Scanner, L2: Stealthy
        self.scan_effort = level_config["scan_effort"] # 스캔 시도 경향 (기본값 1.0)
        self.attack_bias = level_config["attack_bias"] # 지식이 있을 때 공격 시도 경향
        
        # --- [수정] 출력 메시지에 레벨 이름 포함 ---
        print(f"[Seeker] Initialized as {args.seeker_level} ({level_config['name']}): ScanEffort={self.scan_effort}, AttackBias={self.attack_bias}")
        
    def reset(self):
        pass # Env가 Seeker 상태를 모두 관리하므로 reset할 필요 없음

    def select_action(self, mtd_params):
        """
        MTD 파라미터를 기반으로 행동 결정 (scan, exploit, pass)
        """
        
        knowledge = mtd_params["knowledge"]
        
        # 1. 지식이 있고, 공격 성향이 높으면 공격(Exploit) 시도
        if knowledge > 0.1 and random.random() < (self.attack_bias * knowledge):
            return 'exploit'
            
        # 2. 스캔 시도
        # 스캔 노력(scan_effort)이 높을수록, MTD 셔플(ip_cd)이 느릴수록 스캔 시도
        ip_cd_factor = (mtd_params["ip_cd"] / 60.0) # 셔플이 느릴수록(60s) 1.0에 가까워짐
        
        scan_prob = np.clip(self.scan_effort * ip_cd_factor, 0.1, 1.0)
        
        if random.random() < scan_prob:
            return 'scan'

        # 3. 아무것도 안 함
        return 'pass'