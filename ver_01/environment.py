import numpy as np
import random
from config import TESTBED_ACTION_DIM, TESTBED_OBS_DIM

class NetworkEnv:
    def __init__(self, args):
        self.args = args
        
        # --- [핵심] 인터페이스 고정 ---
        self.obs_dim = TESTBED_OBS_DIM
        self.act_dim = TESTBED_ACTION_DIM
        
        # --- MTD 메타 액션 정의 (Config_v21) ---
        self.mtd_action_map = {
            0: ("ip_cd", 1.2),       # Shuffle Faster
            1: ("ip_cd", 0.8),       # Shuffle Slower
            2: ("decoy_ratio", 1.2), # More Decoys
            3: ("decoy_ratio", 0.8), # Less Decoys
            4: ("bl_level", 1.0),    # Increase Block Level
            5: ("bl_level", -1.0),   # Decrease Block Level
            6: ("none", 1.0),        # Pass
        }
        
        # --- MTD 파라미터 범위 정의 (Config_v21) ---
        self.param_ranges = {
            "ip_cd": (args.min_ip_cd, args.max_ip_cd),
            "decoy_ratio": (args.min_decoy_ratio, args.max_decoy_ratio),
            "bl_level": (args.min_bl_level, args.max_bl_level),
        }
        
        # --- 시뮬레이션 상태 변수 ---
        # 1. MTD 파라미터 상태
        self.current_ip_cd = args.base_ip_cd
        self.current_decoy_ratio = args.base_decoy_ratio
        self.current_bl_level = args.base_bl_level
        
        # 2. Attacker 시뮬레이션 상태
        self.attacker_knowledge = 0.0 # 0.0 (모름) ~ 1.0 (완전 파악)
        self.attacker_breached = False
        self.last_attack_type = 'none' # 'scan', 'exploit', 'breach'
        
        # 3. State 벡터용 위협 정보
        self.scan_alert = 0.0
        self.exploit_alert = 0.0
        self.breach_alert = 0.0
        self.mtd_cost_rate = 0.0
        
        self.current_time = 0
        self.seeker = None # (heuristic_seeker.py에서 설정)

    def set_seeker(self, seeker):
        self.seeker = seeker

    def _normalize(self, value, key):
        """파라미터를 0~1 사이로 정규화"""
        min_val, max_val = self.param_ranges[key]
        return (value - min_val) / (max_val - min_val)

    def _get_state(self):
        """
        [핵심] 8차원 상태 벡터 (Testbed-Compatible) 생성
        [ip_cd, decoy_ratio, bl_level, scan, exploit, breach, knowledge, cost]
        """
        state = np.zeros(self.obs_dim)
        
        # 1. 현재 MTD 파라미터 상태 (정규화)
        state[0] = self._normalize(self.current_ip_cd, "ip_cd")
        state[1] = self._normalize(self.current_decoy_ratio, "decoy_ratio")
        state[2] = self._normalize(self.current_bl_level, "bl_level")
        
        # 2. 현재 위협 상태 (지난 스텝의 결과)
        state[3] = self.scan_alert
        state[4] = self.exploit_alert
        state[5] = self.breach_alert
        
        # 3. 기타 상태
        state[6] = self.attacker_knowledge
        state[7] = self.mtd_cost_rate / (self.args.cost_bl_level * 5) # (최대 비용 대비 정규화)
        
        # 다음 스텝을 위해 위협 상태 초기화
        self.scan_alert = 0.0
        self.exploit_alert = 0.0
        self.breach_alert = 0.0
        
        return state

    def reset(self):
        # MTD 파라미터 초기화
        self.current_ip_cd = self.args.base_ip_cd
        self.current_decoy_ratio = self.args.base_decoy_ratio
        self.current_bl_level = self.args.base_bl_level
        
        # Attacker 상태 초기화
        self.attacker_knowledge = 0.0
        self.attacker_breached = False
        self.last_attack_type = 'none'
        
        # State 변수 초기화
        self.scan_alert = 0.0
        self.exploit_alert = 0.0
        self.breach_alert = 0.0
        self.mtd_cost_rate = 0.0
        
        self.current_time = 0
        
        if self.seeker:
            self.seeker.reset()
            
        return self._get_state()

    def _apply_mtd_action(self, mtd_action):
        """MTD 액션을 현재 파라미터에 적용하고 비용 계산"""
        if mtd_action not in self.mtd_action_map:
            return 0.0 # 유효하지 않은 액션

        action_name, value = self.mtd_action_map[mtd_action]
        
        if action_name == "none":
            return self.args.cost_mtd_action # 기본 비용
            
        cost = 0.0
        
        if action_name == "ip_cd":
            self.current_ip_cd = np.clip(
                self.current_ip_cd * value, 
                self.param_ranges["ip_cd"][0], 
                self.param_ranges["ip_cd"][1]
            )
            cost = self.args.cost_shuffle
        
        elif action_name == "decoy_ratio":
            self.current_decoy_ratio = np.clip(
                self.current_decoy_ratio * value, 
                self.param_ranges["decoy_ratio"][0], 
                self.param_ranges["decoy_ratio"][1]
            )
            cost = self.args.cost_decoy_ratio * self.current_decoy_ratio
            
        elif action_name == "bl_level":
            self.current_bl_level = np.clip(
                self.current_bl_level + value, 
                self.param_ranges["bl_level"][0], 
                self.param_ranges["bl_level"][1]
            )
            cost = self.args.cost_bl_level * self.current_bl_level
            
        return cost + self.args.cost_mtd_action

    def _simulate_seeker_step(self):
        """
        현재 MTD 상태를 기반으로 Seeker의 행동과 그 결과를 시뮬레이션
        """
        if not self.seeker:
            return 'NO_SEEKER', 0.0
            
        # Seeker가 현재 MTD 상태를 인지하고 행동 결정
        mtd_params = {
            "ip_cd": self.current_ip_cd,
            "decoy_ratio": self.current_decoy_ratio,
            "bl_level": self.current_bl_level,
            "knowledge": self.attacker_knowledge
        }
        action_type = self.seeker.select_action(mtd_params)
        
        # --- 결과 판정 로직 (Config_v21 기반) ---
        
        if action_type == 'scan':
            self.scan_alert = 1.0
            
            # 1. 디코이에 스캔이 막힘 (decoy_ratio가 높을수록)
            if random.random() < self.current_decoy_ratio:
                return 'HIT_DECOY', self.args.rew_mtd_decoy
                
            # 2. 스캔 성공 (ip_cd가 길수록(느릴수록) 성공률 UP)
            scan_success_prob = (self.current_ip_cd / self.param_ranges["ip_cd"][1])
            if random.random() < scan_success_prob:
                self.attacker_knowledge = min(1.0, self.attacker_knowledge + 0.1) # 지식 획득
                return 'SCAN_SUCCESS', self.args.penalty_mtd_knowledge_leak
            else:
                return 'SCAN_BLOCKED', 0.0 # MTD (빠른 셔플)로 스캔 실패
        
        elif action_type == 'exploit':
            self.exploit_alert = 1.0
            
            # 1. 지식이 부족하여 실패
            if random.random() > self.attacker_knowledge:
                return 'EXPLOIT_FAIL_NO_KNOWLEDGE', self.args.rew_mtd_block_exploit
            
            # 2. Block Level (bl_level)에 의해 차단
            # (v21의 sigmoid 수식을 단순화: bl_level 5일때 90% 차단)
            block_prob = (self.current_bl_level / self.param_ranges["bl_level"][1]) * 0.9
            if random.random() < block_prob:
                return 'EXPLOIT_BLOCKED', self.args.rew_mtd_block_exploit
                
            # 3. Exploit 성공
            # (v21) Exploit 성공 시 Breach 시도
            self.breach_alert = 1.0
            # (v21) Breach 차단 로직 (단순화: bl_level에 비례)
            breach_block_prob = (self.current_bl_level / self.param_ranges["bl_level"][1]) * 0.3
            if random.random() < breach_block_prob:
                return 'BREACH_BLOCKED', self.args.rew_mtd_block_breach + self.args.penalty_mtd_exploit # Exploit은 성공했으나 Breach는 실패
            
            # 4. Breach 최종 성공
            self.attacker_breached = True
            return 'BREACH_SUCCESS', self.args.penalty_mtd_breach + self.args.penalty_mtd_exploit

        return 'PASS', 0.0

    def step(self, mtd_action):
        self.current_time += 1
        
        # 1. Apply MTD Action and get cost
        mtd_cost = self._apply_mtd_action(mtd_action)
        self.mtd_cost_rate = mtd_cost # for state
        
        # 2. Simulate Seeker Step and get outcome/reward
        outcome, outcome_reward = self._simulate_seeker_step()
        
        # 3. Calculate Final Reward
        # MTD 보상/페널티 - MTD 비용
        reward = outcome_reward - (mtd_cost * self.args.cost_weight)
        
        # 4. Check 'done'
        done = False
        if self.attacker_breached:
            done = True
        if self.current_time >= self.args.max_timesteps:
            done = True
            
        # 5. Get next state
        next_state = self._get_state()
            
        return next_state, reward, done, {}

    def close(self):
        pass