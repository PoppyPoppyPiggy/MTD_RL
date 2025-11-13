#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTI Bridge (v2) - argparse compatible
- [MODIFIED] (ver_01) -> (ver_02)
- [MODIFIED] __init__
  'config' (Config object) -> 'args' (argparse namespace)
- [MODIFIED] CTI_Config, REWARDS -> args
- [MODIFIED] State/Action space dimensions from ver_02/config.py
"""

import numpy as np
from config import TESTBED_OBS_DIM, TESTBED_ACTION_DIM

class CTI_Bridge:
    def __init__(self, args):
        print("[CTI Bridge] Initializing (v2, argparse-compatible)...")
        # [MODIFIED] Store args object
        self.args = args

        # [MODIFIED] State/Action space dimensions (ver_02 standard)
        self.mtd_state_dim = TESTBED_OBS_DIM
        self.seeker_state_dim = 6 # (ftp_o, ssh_o, http_o, smb_o, ftp_d, ssh_d)

        # [MODIFIED] State index mapping (ver_02/environment.py 기준)
        self.MTD_STATE_MAP = {
            'threat_level': 0,
            'alert_count': 1,
            'ftp_threat': 2,
            'ssh_threat': 3,
            'http_threat': 4,
            'smb_threat': 5,
            'current_decoy_flag': 6,
            'alert_flag': 7
        }
        self.SEEKER_STATE_MAP = {
            'ftp_open': 0,
            'ssh_open': 1,
            'http_open': 2,
            'smb_open': 3,
            'ftp_decoy': 4,
            'ssh_decoy': 5
        }
        
        self.reset()

    def reset(self):
        self.last_threat_level = 0.0
        self.current_threat_level = 0.0
        self.alert_count = 0
        self.service_threats = {'ftp': 0.0, 'ssh': 0.0, 'http': 0.0, 'smb': 0.0}

    def _get_threat_level_from_obs(self, obs_mtd):
        # ... (기존 코드와 동일) ...
        return 0.0 # (구현 필요)

    def get_mtd_observation(self, state_data):
        # [MODIFIED] This logic must be adapted from mtd_state_reader.py
        # For now, returning a zero vector if state_data is empty
        if not state_data:
             return np.zeros(self.mtd_state_dim)
        
        # (가정) state_data가 mtd_state_reader.py의 get_rl_state()와
        # 동일한 8D 벡터를 'state_vector' 키에 담아준다고 가정
        state_vector = state_data.get('state_vector', np.zeros(self.mtd_state_dim))
        
        # Update internal CTI state based on observation
        self.last_threat_level = self.current_threat_level
        self.current_threat_level = state_vector[self.MTD_STATE_MAP['threat_level']]
        self.alert_count = state_vector[self.MTD_STATE_MAP['alert_count']]
        self.service_threats['ftp'] = state_vector[self.MTD_STATE_MAP['ftp_threat']]
        self.service_threats['ssh'] = state_vector[self.MTD_STATE_MAP['ssh_threat']]
        self.service_threats['http'] = state_vector[self.MTD_STATE_MAP['http_threat']]
        self.service_threats['smb'] = state_vector[self.MTD_STATE_MAP['smb_threat']]
        
        return np.array(state_vector)

    def get_seeker_observation(self, state_data):
        # [MODIFIED] This logic must be adapted from mtd_state_reader.py
        # For now, returning a zero vector if state_data is empty
        if not state_data:
            return np.zeros(self.seeker_state_dim)

        # (가정) state_data가 seeker용 6D 벡터를 'seeker_vector' 키에 담아준다고 가정
        seeker_vector = state_data.get('seeker_vector', np.zeros(self.seeker_state_dim))
        return np.array(seeker_vector)

    def assess_step(self, prev_obs_mtd, prev_obs_seeker, mtd_action, seeker_action, new_obs_mtd, new_obs_seeker):
        
        reward_mtd = 0
        reward_seeker = 0
        done = False

        # --- 1. MTD Cost Calculation ---
        # [MODIFIED] Use args for cost
        mtd_cost = self.args.cost_mtd_action
        if 'decoy' in mtd_action:
            mtd_cost += self.args.cost_mtd_decoy_action
        
        reward_mtd += mtd_cost # Cost is negative reward

        # --- 2. Seeker Action Assessment ---
        is_scan = 'scan' in seeker_action
        is_exploit = 'exploit' in seeker_action
        
        # (Seeker가 스캔/공격한 서비스 타겟)
        target_service = None
        if 'ftp' in seeker_action: target_service = 'ftp'
        elif 'ssh' in seeker_action: target_service = 'ssh'
        elif 'http' in seeker_action: target_service = 'http'
        elif 'smb' in seeker_action: target_service = 'smb'

        if target_service:
            # (Seeker의 관찰 결과)
            was_open = prev_obs_seeker[self.SEEKER_STATE_MAP[f"{target_service}_open"]] > 0
            is_decoy = new_obs_seeker[self.SEEKER_STATE_MAP[f"{target_service}_decoy"]] > 0 # (v2 기준)
            
            if is_scan:
                reward_seeker += -0.1 # (scan cost, hardcoded)
                if was_open:
                    if is_decoy:
                        # Scan Decoy
                        reward_seeker += -0.5
                    else:
                        # Scan Real Target
                        reward_seeker += 0.2
            
            if is_exploit:
                reward_seeker += -0.2 # (exploit cost, hardcoded)
                if was_open:
                    if is_decoy:
                        # Exploit Decoy -> MTD Success
                        reward_mtd += self.args.rew_mtd_decoy
                        reward_seeker += -0.5 # (find decoy)
                    else:
                        # Exploit Real Target -> MTD Fail (Breach)
                        reward_mtd += self.args.penalty_mtd_breach
                        reward_seeker += 1.0 # (breach success)
                        done = True # Breach ends episode
                else:
                    # Exploit closed port
                    reward_seeker += -0.2 # (breach fail)

        # --- 3. MTD Reward Finalization ---
        # (seeker_action이 'noop'이 아닐 때, 즉 CTI가 위협을 감지했을 때)
        is_threat = 'noop' not in seeker_action
        if is_threat:
            # (이전 MTD 상태)
            was_decoy = prev_obs_mtd[self.MTD_STATE_MAP['current_decoy_flag']] > 0
            
            # (현재 MTD 상태)
            is_alert = new_obs_mtd[self.MTD_STATE_MAP['alert_flag']] > 0
            
            # (보상 로직 v21 - MTD가 Decoy를 켰는데, Seeker가 속았을 때)
            # (v23에서는 'Exploit Decoy'에서 이미 처리됨)
            
            # (페널티 로직 v21 - MTD가 Decoy를 안켰는데, Seeker가 스캔 성공 (Leak))
            if is_scan and target_service and not is_decoy and not was_decoy:
                 reward_mtd += self.args.penalty_mtd_breach / 10 # (penalty_mtd_knowledge_leak)

        return reward_mtd, reward_seeker, done, mtd_cost