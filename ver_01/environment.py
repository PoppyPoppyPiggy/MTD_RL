#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Simulation Environment (v22)
- Testbed-Compatible (8D State, 7D Action)
- Simulates Config_v21 parameters (ip_cd, decoy_ratio, bl_level)
- Tracks detailed episode stats for wandb KPI logging
"""

import numpy as np
import random
from config import TESTBED_OBS_DIM, TESTBED_ACTION_DIM
# environment.py 상단 import들 아래에 추가
from typing import Tuple


class NetworkEnv:
    def __init__(self, args, seeker_agent):
        self.args = args
        
        # State/Action Space (테스트베드와 동일하게 고정)
        self.obs_dim = TESTBED_OBS_DIM
        self.act_dim = TESTBED_ACTION_DIM

        # MTD 파라미터 (Config_v21 기준)
        self.param_ranges = {
            "ip_cd": (5.0, 60.0),       # min, max (Shuffle Cooldown)
            "decoy_ratio": (0.0, 0.5),  # min, max
            "bl_level": (0.0, 5.0)      # min, max
        }
        
        # MTD 행동 맵 (테스트베드와 동일하게 고정)
        self.mtd_action_map = {
            0: ("ip_cd", 1.2),       # Faster shuffle (cooldown * 0.8)
            1: ("ip_cd", 0.8),       # Slower shuffle (cooldown * 1.2)
            2: ("decoy_ratio", 1.2), # Increase decoy
            3: ("decoy_ratio", 0.8), # Decrease decoy
            4: ("bl_level", 1.0),    # Increase block level
            5: ("bl_level", -1.0),   # Decrease block level
            6: ("none", 1.0)         # Pass
        }
        
        # 에피소드 상태 변수
        self.current_ip_cd = 30.0
        self.current_decoy_ratio = 0.05
        self.current_bl_level = 1.0
        
        self.attacker_knowledge = 0.0
        self.attacker_breached = False
        
        self.scan_alert = 0.0
        self.exploit_alert = 0.0
        self.breach_alert = 0.0
        self.mtd_cost_rate = 0.0
        
        self.current_time = 0
        self.seeker = seeker_agent
        
        # [KPI] 상세 지표 추적을 위한 통계 딕셔너리
        self.episode_stats = {}
        self.reset() # 초기화 시 _reset_episode_stats 호출

    def set_seeker(self, seeker):
        self.seeker = seeker

    def _get_state(self):
        """테스트베드 호환 8D 상태 벡터 생성"""
        state = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 1. 정규화된 MTD 파라미터 (0.0 ~ 1.0)
        state[0] = (self.current_ip_cd - self.param_ranges["ip_cd"][0]) / (self.param_ranges["ip_cd"][1] - self.param_ranges["ip_cd"][0])
        state[1] = (self.current_decoy_ratio - self.param_ranges["decoy_ratio"][0]) / (self.param_ranges["decoy_ratio"][1] - self.param_ranges["decoy_ratio"][0])
        state[2] = (self.current_bl_level - self.param_ranges["bl_level"][0]) / (self.param_ranges["bl_level"][1] - self.param_ranges["bl_level"][0])
        
        # 2. 위협 상태 (0.0 or 1.0)
        state[3] = self.scan_alert
        state[4] = self.exploit_alert
        state[5] = self.breach_alert
        
        # 3. 공격자 지식 (0.0 ~ 1.0)
        state[6] = self.attacker_knowledge
        
        # 4. MTD 비용 (정규화)
        max_cost = (self.args.cost_shuffle * self.param_ranges["ip_cd"][1]) + \
                   (self.args.cost_decoy_ratio * self.param_ranges["decoy_ratio"][1]) + \
                   (self.args.cost_bl_level * self.param_ranges["bl_level"][1])
        state[7] = self.mtd_cost_rate / (max_cost + 1e-6)
        
        # 다음 스텝을 위해 위협 상태 초기화
        self.scan_alert = 0.0
        self.exploit_alert = 0.0
        self.breach_alert = 0.0
        self.mtd_cost_rate = 0.0
        
        return state

    def _reset_episode_stats(self):
        """[KPI] 에피소드 시작 시 통계 딕셔너리를 0으로 초기화"""
        self.episode_stats = {
            "time_to_breach": self.args.max_timesteps, # Breach 안 당하면 최대 시간
            "time_to_first_alert": self.args.max_timesteps, # 탐지 못하면 최대 시간
            "first_alert_triggered": False,
            
            # MTD 행동 카운트
            "mtd_actions_taken": 0,     # 'none'이 아닌 MTD 행동 총 횟수
            "mtd_shuffle_actions": 0,   # ip_cd (0, 1)
            "mtd_decoy_actions": 0,     # decoy_ratio (2, 3)
            "mtd_bl_actions": 0,        # bl_level (4, 5)
            "mtd_pass_actions": 0,      # 'none' (6)
            
            # Seeker 행동 카운트
            "seeker_scans": 0,
            "seeker_exploits": 0,
            
            # Outcome 카운트
            "outcome_scan_success": 0,
            "outcome_scan_hit_decoy": 0,
            "outcome_scan_blocked": 0,    # [신규] 셔플로 인한 스캔 차단
            "outcome_exploit_blocked": 0,
            "outcome_breach_blocked": 0,
            "outcome_breach_success": 0,

            # 최종 정책 값 (KPI)
            "final_ip_cd": self.current_ip_cd,
            "final_decoy_ratio": self.current_decoy_ratio,
            "final_bl_level": self.current_bl_level,
            "final_attacker_knowledge": 0.0,
        }

    def reset(self):
        # MTD 파라미터 초기화
        self.current_ip_cd = 30.0
        self.current_decoy_ratio = 0.05
        self.current_bl_level = 1.0
        
        self.attacker_knowledge = 0.0
        self.attacker_breached = False
        
        self.mtd_cost_rate = 0.0
        self.current_time = 0
        
        # [KPI] 통계 초기화 호출
        self._reset_episode_stats()
        
        if self.seeker:
            self.seeker.reset()
            
        return self._get_state()

    def _apply_mtd_action(self, mtd_action: int) -> float:
        """MTD 행동을 적용하고 비용(cost)을 반환"""
        action_name, value = self.mtd_action_map[mtd_action]
        
        if action_name == "none":
            self.episode_stats["mtd_pass_actions"] += 1
            return self.args.cost_mtd_action # 기본 비용
            
        cost = 0.0
        self.episode_stats["mtd_actions_taken"] += 1
        
        if action_name == "ip_cd":
            # 0: 쿨다운 감소 (더 빠른 셔플), 1: 쿨다운 증가 (더 느린 셔플)
            new_val = self.current_ip_cd / value if mtd_action == 0 else self.current_ip_cd * value
            self.current_ip_cd = np.clip(new_val, self.param_ranges["ip_cd"][0], self.param_ranges["ip_cd"][1])
            cost = self.args.cost_shuffle * (self.param_ranges["ip_cd"][1] - self.current_ip_cd) # 셔플 빈도(비용)는 쿨다운과 반비례
            self.episode_stats["mtd_shuffle_actions"] += 1
        
        elif action_name == "decoy_ratio":
            new_val = self.current_decoy_ratio * value if mtd_action == 2 else self.current_decoy_ratio / value
            self.current_decoy_ratio = np.clip(new_val, self.param_ranges["decoy_ratio"][0], self.param_ranges["decoy_ratio"][1])
            cost = self.args.cost_decoy_ratio * self.current_decoy_ratio
            self.episode_stats["mtd_decoy_actions"] += 1
            
        elif action_name == "bl_level":
            new_val = self.current_bl_level + value
            self.current_bl_level = np.clip(new_val, self.param_ranges["bl_level"][0], self.param_ranges["bl_level"][1])
            cost = self.args.cost_bl_level * self.current_bl_level
            self.episode_stats["mtd_bl_actions"] += 1
            
        return cost + self.args.cost_mtd_action

    def _simulate_seeker_action(self) -> Tuple[str, float]:
        """Seeker의 행동을 시뮬레이션하고 (결과, 보상)을 반환"""
        
        # Seeker가 현재 MTD 파라미터를 인지
        mtd_params = {
            "ip_cd": self.current_ip_cd,
            "decoy_ratio": self.current_decoy_ratio,
            "bl_level": self.current_bl_level
        }
        action_type = self.seeker.select_action(self.attacker_knowledge, mtd_params)
        
        # [KPI] 최초 탐지 시간 기록
        if action_type in ['scan', 'exploit'] and not self.episode_stats["first_alert_triggered"]:
            self.episode_stats["time_to_first_alert"] = self.current_time
            self.episode_stats["first_alert_triggered"] = True

        if action_type == 'scan':
            self.scan_alert = 1.0
            self.episode_stats["seeker_scans"] += 1
            
            # 1. 디코이에 스캔이 막힘 (decoy_ratio가 높을수록)
            if random.random() < self.current_decoy_ratio:
                self.episode_stats["outcome_scan_hit_decoy"] += 1
                return 'HIT_DECOY', self.args.rew_mtd_decoy
                
            # 2. 셔플(ip_cd)로 인해 스캔이 막힘 (ip_cd가 낮을수록(빠를수록) 방어 확률 UP)
            # ip_cd를 0~1로 정규화 (낮을수록 좋음)
            ip_cd_norm = (self.current_ip_cd - self.param_ranges["ip_cd"][0]) / (self.param_ranges["ip_cd"][1] - self.param_ranges["ip_cd"][0])
            scan_block_prob = (1.0 - ip_cd_norm) * 0.5 # 쿨다운이 길면(norm=1) 방어 0%, 짧으면(norm=0) 방어 50%
            
            if random.random() < scan_block_prob:
                 self.episode_stats["outcome_scan_blocked"] += 1
                 return 'SCAN_BLOCKED', 0.0 # 스캔은 막았지만 보상은 없음 (디코이와 차별)

            # 3. 스캔 성공 (디코이와 셔플을 모두 통과)
            self.attacker_knowledge = min(1.0, self.attacker_knowledge + 0.1) # 지식 획득
            self.episode_stats["outcome_scan_success"] += 1
            return 'SCAN_SUCCESS', self.args.penalty_mtd_knowledge_leak
        
        elif action_type == 'exploit':
            self.exploit_alert = 1.0
            self.episode_stats["seeker_exploits"] += 1
            
            # 1. 지식이 부족하여 실패
            if random.random() > self.attacker_knowledge:
                return 'EXPLOIT_FAIL_NO_KNOWLEDGE', 0.0 # MTD의 공이 아님
            
            # 2. Exploit이 MTD(bl_level)에 의해 차단됨
            # bl_level을 0~1로 정규화
            bl_norm = (self.current_bl_level - self.param_ranges["bl_level"][0]) / (self.param_ranges["bl_level"][1] - self.param_ranges["bl_level"][0])
            exploit_block_prob = bl_norm * 0.9 # 최대 90% 확률로 차단 (Config_v21의 BLOCK_LOUD_K 근사)
            
            if random.random() < exploit_block_prob:
                self.episode_stats["outcome_exploit_blocked"] += 1
                return 'EXPLOIT_BLOCKED', self.args.rew_mtd_block_exploit
                
            # 3. Exploit 성공, Breach 시도 (Config_v21의 EXPLOITED_BREACH_P=0.9)
            if random.random() < 0.9:
                self.breach_alert = 1.0
                
                # 3a. Breach가 MTD(bl_level)에 의해 차단됨
                breach_block_prob = bl_norm * 0.3 # 최대 30% 확률로 차단 (Config_v21의 BLOCK_BREACH_K 근사)
                if random.random() < breach_block_prob:
                    self.episode_stats["outcome_breach_blocked"] += 1
                    return 'BREACH_BLOCKED', self.args.rew_mtd_block_breach + self.args.penalty_mtd_exploit # Exploit은 성공(페널티)했으나 Breach는 실패(보상)
            
                # 3b. Breach 최종 성공
                self.attacker_breached = True
                self.episode_stats["outcome_breach_success"] += 1
                return 'BREACH_SUCCESS', self.args.penalty_mtd_breach + self.args.penalty_mtd_exploit
            else:
                # 3c. Exploit은 성공했으나, Breach 시도 자체를 실패 (MTD 공 아님)
                return 'EXPLOIT_SUCCESS_BREACH_FAIL', self.args.penalty_mtd_exploit

        return 'PASS', 0.0 # Seeker가 아무것도 안 함

    def step(self, mtd_action):
        self.current_time += 1
        
        info = {} # [KPI] info 딕셔너리 초기화
        
        # 1. Apply MTD Action and get cost
        mtd_cost = self._apply_mtd_action(mtd_action)
        self.mtd_cost_rate = mtd_cost # (다음 state를 위해 비용 저장)
        
        # 2. Simulate Seeker Action and get (outcome, reward)
        outcome, outcome_reward = self._simulate_seeker_action()
        
        # 3. Calculate final reward (Reward - Cost)
        reward = outcome_reward - (mtd_cost * self.args.cost_weight)
        
        # 4. Check 'done'
        done = False
        if self.attacker_breached:
            done = True
            # [KPI] Breach 성공 시, 시간 기록
            self.episode_stats["time_to_breach"] = self.current_time
            
        if self.current_time >= self.args.max_timesteps:
            done = True
            
        # 5. Get next state
        next_state = self._get_state()
        
        # [KPI] 에피소드가 끝나면 최종 통계 info에 추가
        if done:
            self.episode_stats["final_attacker_knowledge"] = self.attacker_knowledge
            self.episode_stats["final_ip_cd"] = self.current_ip_cd
            self.episode_stats["final_decoy_ratio"] = self.current_decoy_ratio
            self.episode_stats["final_bl_level"] = self.current_bl_level
            info['episode_stats'] = self.episode_stats
            
        return next_state, reward, float(done), info

    def close(self):
        print("Closing Environment...")