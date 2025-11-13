#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testbed-linked MTD Environment (v2)
- MTD_RL/ver_02
- [MODIFIED] Added detailed info keys to the step() function
  to support mtd_scoring.py logic calculation in the training loop.
"""

import os
import json
import time
import numpy as np
import redis
from config import Config
from seeker import Seeker  # Seeker (Attacker)
from heuristic_seeker import HeuristicSeeker
# CTI Bridge (from ver_01)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ver_01'))
from cti_bridge import CTI_Bridge

class MTDEnv:
    def __init__(self, config: Config):
        print("Initializing MTDEnv (v2)...")
        self.config = config
        self.env_config = config.ENV
        
        # Action/State space dimensions
        self.mtd_action_dim = self.env_config['mtd_action_dim']
        self.mtd_state_dim = self.env_config['mtd_state_dim']
        self.seeker_action_dim = self.env_config['seeker_action_dim']
        self.seeker_state_dim = self.env_config['seeker_state_dim']
        
        # Action maps (Index -> Name)
        self.mtd_action_map = self.env_config['mtd_action_map']
        self.seeker_action_map = self.env_config['seeker_action_map']
        
        # State (Observation)
        self.obs_mtd = np.zeros(self.mtd_state_dim)
        self.obs_seeker = np.zeros(self.seeker_state_dim)
        
        # CTI Bridge
        self.cti_bridge = CTI_Bridge(config.CTI_Config)
        
        # Seeker (Attacker)
        if self.config.SEEKER_CONFIG['strategy'] == 'heuristic':
            print("Using HeuristicSeeker")
            self.seeker = HeuristicSeeker(self.seeker_state_dim, self.seeker_action_dim, config.SEEKER_CONFIG)
        else:
            print(f"Using RL Seeker (Policy: {self.config.SEEKER_CONFIG['policy_path']})")
            self.seeker = Seeker(self.seeker_state_dim, self.seeker_action_dim, config.SEEKER_CONFIG)
            self.seeker.load(self.config.SEEKER_CONFIG['policy_path'])
            
        # Redis connection
        try:
            self.redis_db = redis.StrictRedis(host=self.config.REDIS_HOST, port=self.config.REDIS_PORT, db=0, decode_responses=True)
            self.redis_db.ping()
            print(f"Redis connected at {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            sys.exit(1)
            
        self.state_sub = self.redis_db.pubsub(ignore_subscribe_messages=True)
        self.state_sub.subscribe(self.config.STATE_CHANNEL)
        self.last_mtd_action_id = -1
        self._wait_for_initial_state()

    def _wait_for_initial_state(self):
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
        self.redis_db.publish(self.config.ACTION_CHANNEL, json.dumps({"command": "RESET"}))
        self.cti_bridge.reset()
        self.seeker.reset()
        self._wait_for_initial_state()
        return self.obs_mtd

    def step(self, mtd_action: int):
        """
        Perform one environment step.
        1. (MTD) Publish MTD action to testbed.
        2. (Seeker) Get Seeker action based on current state.
        3. (Testbed) Wait for the *next* state from testbed (which results from MTD+Seeker actions).
        4. (CTI) Assess the new state to calculate rewards.
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
        self.redis_db.publish(self.config.ACTION_CHANNEL, json.dumps(action_payload))
        
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
                                     'move' not in seeker_action_name and \
                                     'noop' not in seeker_action_name
        # 'is_decoy_action': Any MTD action that involves a decoy
        info['is_decoy_action'] = 'decoy' in mtd_action_name 

        # Attack Resilience (R_A) metrics
        # 'is_breach': Seeker reward is high (successfully exploited a non-decoy)
        # We check if the seeker reward is high (e.g., > 0.8, proxy for +1.0)
        # AND the MTD action was NOT a decoy (meaning a real asset was hit)
        is_breach_success = reward_seeker > 0.8
        info['is_breach'] = is_breach_success and not info['is_decoy_action']

        return self.obs_mtd, reward_mtd, done, info