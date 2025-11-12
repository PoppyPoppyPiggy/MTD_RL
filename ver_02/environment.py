# File: ver_02/environment.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Simulation Environment (v23 - 배포 환경 호환)

[수정]
이 시뮬레이터는 'rl_driven_deception_manager.py' 및 'iptables_mtd.yaml'과
동일한 State/Action 공간을 시뮬레이션합니다.

- Action (7D): [DNAT_R1, DNAT_R2, ..., DNAT_R6, DNAT_Decoy] (ID 0~6)
- State (8D):  [R1_active, R2_active, ..., Decoy_active, Alert_active] (One-hot)
"""

import numpy as np
import random
from config import TESTBED_OBS_DIM, TESTBED_ACTION_DIM
from typing import Tuple

class NetworkEnv:
    def __init__(self, args, seeker_agent):
        self.args = args
        
        # --- [수정] 배포 환경(Testbed)과 동일한 State/Action 공간 ---
        self.obs_dim = TESTBED_OBS_DIM    # 8 (6 Real + 1 Decoy + 1 Alert)
        self.act_dim = TESTBED_ACTION_DIM # 7 (Action 0~6)
        
        if self.obs_dim != 8 or self.act_dim != 7:
            print(f"Warning: config.py의 차원(OBS:{self.obs_dim}, ACT:{self.act_dim})이 배포 환경(8, 7)과 다릅니다.")

        # --- 시뮬레이션 내부 변수 ---
        self.seeker = seeker_agent
        self.current_time = 0
        
        # MTD 상태 (Action ID 0~6이 현재 활성 타겟을 의미)
        # ID 0-5: Real Targets (R1~R6)
        # ID 6: Decoy
        self.active_mtd_id = 6 # 초기 상태는 Decoy(ID 6)라고 가정
        
        self.alert_active = 0.0 # 현재 스텝에서 CTI 알림이 발생했는지 (0.0 또는 1.0)
        
        self.attacker_knowledge = 0.0 # (내부용) Seeker의 지식 수준
        self.attacker_breached = False
        
        # [KPI] 상세 지표 추적
        self.episode_stats = {}
        self._reset_episode_stats() # episode_stats 딕셔너리 초기화

    def set_seeker(self, seeker):
        self.seeker = seeker

    def _get_state(self) -> np.ndarray:
        """
        [수정] 배포 환경(mtd_state_reader.py)과 동일한 8D 상태 벡터 생성
        [R1_active, R2_active, ..., R6_active, Decoy_active, Alert_active]
        """
        state = np.zeros(self.obs_dim, dtype=np.float32)
        
        # 1. State[0-6]: One-hot 인코딩된 현재 활성 MTD
        #    active_mtd_id가 0~6 사이의 값이므로, 해당 인덱스에 1.0을 설정
        if 0 <= self.active_mtd_id < self.act_dim: # 0~6
            state[self.active_mtd_id] = 1.0
        
        # 2. State[7]: CTI 위협 알림 (0.0 또는 1.0)
        state[self.obs_dim - 1] = self.alert_active
        
        # 3. 다음 스텝을 위해 알림 상태 초기화
        self.alert_active = 0.0
        
        return state

    def _reset_episode_stats(self):
        """[KPI] 에피소드 시작 시 통계 딕셔너리를 0으로 초기화"""
        self.episode_stats = {
            "time_to_breach": self.args.max_timesteps, # Breach 안 당하면 최대 시간
            "time_to_first_alert": self.args.max_timesteps, # 탐지 못하면 최대 시간
            "first_alert_triggered": False,
            
            # MTD 행동 카운트 (0~6)
            "mtd_action_counts": [0] * self.act_dim, # 7개 행동 각각의 카운트
            
            # Seeker 행동 카운트
            "seeker_scans": 0,
            "seeker_exploits": 0,
            
            # Outcome 카운트
            "outcome_scan_success": 0,
            "outcome_scan_hit_decoy": 0,
            "outcome_scan_failed": 0, # (지식 부족 등)
            "outcome_exploit_blocked": 0, # (Seeker가 Exploit 시도 -> MTD가 막음)
            "outcome_exploit_success": 0, # (Seeker가 Exploit 시도 -> MTD가 못막음)
            "outcome_breach_success": 0, # (Exploit 성공 후 Breach까지 성공)

            # 최종 상태 값
            "final_attacker_knowledge": 0.0,
            "final_active_mtd_id": self.active_mtd_id
        }

    def reset(self):
        # MTD 상태 초기화 (Decoy에서 시작)
        self.active_mtd_id = 6 # ID 6 = Decoy
        self.alert_active = 0.0
        
        self.attacker_knowledge = 0.0
        self.attacker_breached = False
        self.current_time = 0
        
        # [KPI] 통계 초기화
        self._reset_episode_stats()
        
        if self.seeker:
            self.seeker.reset()
            
        return self._get_state()

    def _simulate_seeker_action(self) -> Tuple[str, float]:
        """Seeker의 행동을 시뮬레이션하고 (결과, 보상)을 반환"""
        
        # Seeker가 현재 MTD 상태(ID 0~6)를 인지
        # (HeuristicSeeker.py(v23)가 이 mtd_id를 보고 행동을 결정)
        seeker_action_type = self.seeker.select_action(self.attacker_knowledge, self.active_mtd_id)
        
        # [KPI] 최초 탐지 시간 기록
        if seeker_action_type in ['scan', 'exploit'] and not self.episode_stats["first_alert_triggered"]:
            self.episode_stats["time_to_first_alert"] = self.current_time
            self.episode_stats["first_alert_triggered"] = True

        if seeker_action_type == 'scan':
            self.alert_active = 1.0 # CTI 알림 발생
            self.episode_stats["seeker_scans"] += 1
            
            # 1. MTD가 Decoy(ID 6)를 활성화한 상태
            if self.active_mtd_id == 6:
                self.episode_stats["outcome_scan_hit_decoy"] += 1
                return 'HIT_DECOY', self.args.rew_mtd_decoy # (보상)
            
            # 2. MTD가 Real Target(ID 0~5)을 활성화한 상태
            else:
                self.attacker_knowledge = min(1.0, self.attacker_knowledge + 0.1) # 지식 획득
                self.episode_stats["outcome_scan_success"] += 1
                return 'SCAN_SUCCESS', self.args.penalty_mtd_knowledge_leak # (페널티)
        
        elif seeker_action_type == 'exploit':
            self.alert_active = 1.0 # CTI 알림 발생
            self.episode_stats["seeker_exploits"] += 1
            
            # 1. 지식이 부족하여 실패
            if random.random() > self.attacker_knowledge:
                self.episode_stats["outcome_scan_failed"] += 1 # (Exploit이 Scan처럼 실패)
                return 'EXPLOIT_FAIL_NO_KNOWLEDGE', 0.0 # MTD의 공이 아님
            
            # 2. MTD가 Decoy(ID 6)를 활성화한 상태
            if self.active_mtd_id == 6:
                self.episode_stats["outcome_exploit_blocked"] += 1
                # [v23] rew_mtd_block_exploit는 rew_mtd_decoy와 같음
                return 'EXPLOIT_BLOCKED', self.args.rew_mtd_block_exploit # (보상)
            
            # 3. MTD가 Real Target(ID 0~5)을 활성화한 상태 (공격 성공)
            else:
                self.attacker_breached = True # [!!!] 시스템 침투 성공
                self.episode_stats["outcome_exploit_success"] += 1
                self.episode_stats["outcome_breach_success"] += 1
                return 'BREACH_SUCCESS', self.args.penalty_mtd_breach # (치명적 페널티)

        return 'PASS', 0.0 # Seeker가 아무것도 안 함

    def step(self, mtd_action: int):
        self.current_time += 1
        info = {}
        
        # 1. MTD 행동(0~6) 적용
        # MTD 행동 자체에 대한 비용 (e.g., 10초마다 '고민'하는 비용)
        mtd_cost = self.args.cost_mtd_action 
        
        # [KPI] MTD 행동 카운트
        if 0 <= mtd_action < self.act_dim:
            self.episode_stats["mtd_action_counts"][mtd_action] += 1
        
        # MTD 행동(타겟 변경)이 이전과 다를 경우 추가 비용
        if self.active_mtd_id != mtd_action:
             mtd_cost += self.args.cost_shuffle # (cost_shuffle을 '변경 비용'으로 재정의)
             
        self.active_mtd_id = mtd_action # MTD 상태(활성 타겟) 업데이트

        # 2. Seeker 행동 시뮬레이션 및 (결과, 보상) 획득
        outcome, outcome_reward = self._simulate_seeker_action()
        
        # 3. 최종 보상 계산 (보상 - 비용)
        reward = outcome_reward - (mtd_cost * self.args.cost_weight)
        
        # 4. 종료 조건 확인
        done = False
        if self.attacker_breached:
            done = True
            # [KPI] Breach 성공 시, 시간 기록
            self.episode_stats["time_to_breach"] = self.current_time
            
        if self.current_time >= self.args.max_timesteps:
            done = True
            
        # 5. 다음 상태(State) 가져오기
        next_state = self._get_state()
        
        # [KPI] 에피소드가 끝나면 최종 통계 info에 추가
        if done:
            self.episode_stats["final_attacker_knowledge"] = self.attacker_knowledge
            self.episode_stats["final_active_mtd_id"] = self.active_mtd_id
            info['episode_stats'] = self.episode_stats.copy() # [FIX] 복사본 전달
            
        return next_state, reward, float(done), info

    def close(self):
        print("Closing Environment...")