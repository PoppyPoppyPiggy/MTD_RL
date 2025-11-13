#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testbed-linked MTD Environment (v2)
- MTD_RL/ver_02
- [MODIFIED] Added detailed info keys to the step() function
  to support mtd_scoring.py logic calculation in the training loop.
- [MODIFIED] __init__ updated to accept 'args' from argparse (ver_03 style)
"""

import os
import json
import time
import numpy as np
import redis
# [MODIFIED] Import get_seeker_policy_path and TESTBED dimensions
from config import get_seeker_policy_path, TESTBED_OBS_DIM, TESTBED_ACTION_DIM
from seeker import SeekerHands  # Seeker (Attacker)
from heuristic_seeker import HeuristicSeeker
# CTI Bridge (from ver_01)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ver_01'))
from cti_bridge import CTI_Bridge

class MTDEnv:
    # [MODIFIED] __init__ now accepts 'args' from argparse
    def __init__(self, args):
        print("Initializing MTDEnv (v2, argparse-compatible)...")
        self.args = args # Store args
        
        # Action/State space dimensions from config.py
        self.mtd_action_dim = TESTBED_ACTION_DIM
        self.mtd_state_dim = TESTBED_OBS_DIM
        self.seeker_action_dim = 6 # (scan_ftp, ssh, http, smb, exploit, noop)
        self.seeker_state_dim = 6 # (ftp_o, ssh_o, http_o, smb_o, ftp_d, ssh_d)
        
        # Action maps (Index -> Name)
        self.mtd_action_map = {
            0: 'service_shuffle', # (ver_02/config.py 기준)
            1: 'port_shuffle',
            2: 'decoy_ftp',
            3: 'decoy_ssh',
            4: 'decoy_http',
            5: 'decoy_smb',
            6: 'noop'
        }
        self.seeker_action_map = {
            0: 'scan_ftp',
            1: 'scan_ssh',
            2: 'scan_http',
            3: 'scan_smb',
            4: 'exploit',
            5: 'noop'
        }
        
        # State (Observation)
        self.obs_mtd = np.zeros(self.mtd_state_dim)
        self.obs_seeker = np.zeros(self.seeker_state_dim)
        
        # CTI Bridge
        # [MODIFIED] Pass args to CTI_Bridge
        self.cti_bridge = CTI_Bridge(self.args) 
        
        # Seeker (Attacker)
        # [MODIFIED] Use args.seeker_level to configure Seeker
        seeker_policy_path = get_seeker_policy_path(args.seeker_level)
        
        if args.seeker_level == "L0" or seeker_policy_path is None:
            print("Using HeuristicSeeker (L0 or policy file not found)")
            self.seeker_strategy = 'heuristic'
            self.seeker = HeuristicSeeker(self.seeker_state_dim, self.seeker_action_dim, self.args)
        else:
            print(f"Using RL Seeker (Policy: {seeker_policy_path})")
            self.seeker_strategy = 'rl'
            self.seeker = Seeker(self.seeker_state_dim, self.seeker_action_dim, self.args)
            self.seeker.load(seeker_policy_path)
            
        # Redis connection (Hardcoded, as in original ver_02 config)
        self.redis_host = '127.0.0.1'
        self.redis_port = 6379
        self.state_channel = 'mtd_state_channel'
        self.action_channel = 'mtd_action_channel'
        
        try:
            self.redis_db = redis.StrictRedis(host=self.redis_host, port=self.redis_port, db=0, decode_responses=True)
            self.redis_db.ping()
            print(f"Redis connected at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            sys.exit(1)
            
        self.state_sub = self.redis_db.pubsub(ignore_subscribe_messages=True)
        self.state_sub.subscribe(self.state_channel)
        self.last_mtd_action_id = -1
        self._wait_for_initial_state()

    def _wait_for_initial_state(self):
        # ... (기존 코드와 동일) ...
        print("Waiting for initial state from testbed...")
        while True:
            msg = self.state_sub.get_message(timeout=5.0)
            if msg:
                state_data = json.loads(msg['data'])
                self._update_obs_from_state(state_data)
                print("Initial state received.")
                break
            else:
                print("...still waiting for state...")

    def _update_obs_from_state(self, state_data):
        """Helper to update internal observations based on full state data"""
        # MTD Observation (Threat-centric)
        self.obs_mtd = self.cti_bridge.get_mtd_observation(state_data)
        
        # Seeker Observation (Network-centric)
        self.obs_seeker = self.cti_bridge.get_seeker_observation(state_data)

    def reset(self):
        """Resets the environment by sending a RESET command"""
        print("Environment reset requested.")
        self.redis_db.publish(self.action_channel, json.dumps({"command": "RESET"}))
        self.cti_bridge.reset()
        self.seeker.reset()
        self._wait_for_initial_state()
        return self.obs_mtd

    def step(self, mtd_action: int):
        """
        Perform one environment step.
        ... (기존 설명과 동일) ...
        5. (Scoring) [MODIFIED] Return detailed info for mtd_scoring calculation.
        """
        
        # 1. (MTD) Publish MTD action
        mtd_action_name = self.mtd_action_map[mtd_action]
        mtd_action_id = (self.last_mtd_action_id + 1) % 10000 # Simple incrementing ID
        self.last_mtd_action_id = mtd_action_id
        
        action_payload = {
            "command": "RUN_MTD",
            "action_name": mtd_action_name,
            "action_id": mtd_action_id
        }
        self.redis_db.publish(self.action_channel, json.dumps(action_payload))
        
        # 2. (Seeker) Get Seeker action
        seeker_action = self.seeker.act(self.obs_seeker)
        seeker_action_name = self.seeker_action_map[seeker_action]

        # 3. (Testbed) Wait for the next state
        next_state_data = None
        while True:
            msg = self.state_sub.get_message(timeout=10.0)
            if msg:
                next_state_data = json.loads(msg['data'])
                break
            else:
                print("Warning: Timeout waiting for next state in step(). Returning done=True.")
                return self.obs_mtd, 0, True, {} # Timeout
        
        # 4. (CTI) Assess new state and update observations
        prev_obs_mtd = self.obs_mtd
        prev_obs_seeker = self.obs_seeker
        self._update_obs_from_state(next_state_data) # self.obs_mtd and self.obs_seeker are updated here
        
        # [MODIFIED] CTI Bridge가 args를 기반으로 보상 계산
        reward_mtd, reward_seeker, done, mtd_cost = self.cti_bridge.assess_step(
            prev_obs_mtd=prev_obs_mtd,
            prev_obs_seeker=prev_obs_seeker,
            mtd_action=mtd_action_name,
            seeker_action=seeker_action_name,
            new_obs_mtd=self.obs_mtd,
            new_obs_seeker=self.obs_seeker
        )
        
        # 5. [MODIFIED] Prepare detailed info dictionary for scoring
        info = {}
        
        # Cost metric (C_M)
        info['cost'] = mtd_cost 
        
        # Deception Success (S_D) metrics
        # 'is_attack_detected': Any seeker action that isn't reconnaissance
        info['is_attack_detected'] = 'scan' not in seeker_action_name and \
                                     'noop' not in seeker_action_name
        # 'is_decoy_action': Any MTD action that involves a decoy
        info['is_decoy_action'] = 'decoy' in mtd_action_name 

        # Attack Resilience (R_A) metrics
        # 'is_breach': Seeker reward is high (successfully exploited a non-decoy)
        # We check if the seeker reward is high (e.g., > 0.8, proxy for +1.0)
        # AND the MTD action was NOT a decoy (meaning a real asset was hit)
        
        # [MODIFIED] ver_03/config.py의 보상 체계(penalty_mtd_breach)에 맞게 수정
        # 시커 보상이 0보다 크면(성공) 디코이가 아니었을 때 Breach로 간주
        is_breach_success = reward_seeker > 0 
        info['is_breach'] = is_breach_success and not info['is_decoy_action']

        return self.obs_mtd, reward_mtd, done, info