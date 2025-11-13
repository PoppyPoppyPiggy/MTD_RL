# ver_03/heuristic_seeker.py
# 'rl/ver_02'의 heuristic_seeker.py와 동일. (ver_03용으로 복사)
# ver_03의 environment.py가 seeker의 행동을 시뮬레이션하므로,
# 이 파일은 직접 사용되지는 않지만 참조용 또는 향후 연동을 위해 유지.

import numpy as np

class HeuristicSeeker:
    """
    A more advanced heuristic seeker that models different levels of knowledge
    and scanning strategies (L0 to L4).
    """
    def __init__(self, n_targets, n_decoys, seeker_level=0):
        self.n_targets = n_targets
        self.n_decoys = n_decoys
        self.total_nodes = n_targets + n_decoys
        self.seeker_level = seeker_level # L0: Naive, L4: Advanced

        # 0: unknown, 1: known_exist, 2: known_decoy, 3: known_real
        self.knowledge = np.zeros(self.total_nodes)
        self.scan_history = []
        self.exploit_target = -1
        
        # Seeker scanning strategy
        self.scan_idx = 0

    def reset(self):
        self.knowledge = np.zeros(self.total_nodes)
        self.scan_history = []
        self.exploit_target = -1
        self.scan_idx = 0

    def _get_scan_target(self):
        """Determine which node to scan next based on level."""
        if self.seeker_level == 0: # Naive: Scan sequentially
            target = self.scan_idx
            self.scan_idx = (self.scan_idx + 1) % self.total_nodes
            return target
        
        else: # L1+: Prioritize unknown nodes
            unknown_nodes = np.where(self.knowledge == 0)[0]
            if len(unknown_nodes) > 0:
                return np.random.choice(unknown_nodes)
            else:
                # All nodes known, rescan known_exist
                known_exist = np.where(self.knowledge == 1)[0]
                if len(known_exist) > 0:
                    return np.random.choice(known_exist)
                else:
                    # Scan randomly
                    return np.random.randint(0, self.total_nodes)

    def scan(self, env_state):
        """
        Simulates scanning the environment.
        env_state: [target_0, ..., target_N, decoy_0, ..., decoy_M, cti]
        """
        scan_target_idx = self._get_scan_target()
        
        is_active = env_state[scan_target_idx]
        is_decoy = scan_target_idx >= self.n_targets
        
        if self.seeker_level < 2: # L0/L1: Can't distinguish real/decoy
            if is_active:
                self.knowledge[scan_target_idx] = 1 # known_exist
                if self.exploit_target == -1:
                    self.exploit_target = scan_target_idx
            else:
                self.knowledge[scan_target_idx] = 0 # unknown (or mark as inactive)

        else: # L2+: Can distinguish real/decoy (if CTI is high or advanced scan)
            cti_state = env_state[-1]
            can_distinguish = (self.seeker_level >= 3) or (cti_state == 1 and self.seeker_level == 2)

            if is_active:
                if can_distinguish:
                    if is_decoy:
                        self.knowledge[scan_target_idx] = 2 # known_decoy
                    else:
                        self.knowledge[scan_target_idx] = 3 # known_real
                        if self.exploit_target == -1 or self.knowledge[self.exploit_target] != 3:
                            self.exploit_target = scan_target_idx # Prioritize real target
                else:
                    self.knowledge[scan_target_idx] = 1 # known_exist
                    if self.exploit_target == -1:
                        self.exploit_target = scan_target_idx
            else:
                 self.knowledge[scan_target_idx] = 0 # unknown

    def exploit(self, env_state):
        """Tries to exploit a known target."""
        
        # L4: Only exploits targets known as 'real'
        if self.seeker_level == 4 and self.exploit_target != -1:
            if self.knowledge[self.exploit_target] != 3:
                self.exploit_target = -1 # Refuses to exploit non-real targets

        # Find a new target if current is invalid
        if self.exploit_target == -1:
            priority_targets = np.where(self.knowledge == 3)[0] # L3/L4 look for real
            if len(priority_targets) == 0 and self.seeker_level < 4:
                priority_targets = np.where(self.knowledge == 1)[0] # L0-L3 look for any
            
            if len(priority_targets) > 0:
                self.exploit_target = np.random.choice(priority_targets)
            else:
                return 'scan' # No known targets to exploit

        # Try exploiting
        if env_state[self.exploit_target] == 1:
            return 'success' # Exploit successful
        else:
            # Exploit failed, target moved or was decoy
            self.knowledge[self.exploit_target] = 0 # Mark as unknown
            self.exploit_target = -1
            return 'fail_rescan' # Must find a new target

    def choose_action(self, env_state):
        """Heuristic action choice based on level."""
        
        # Exploit vs Scan
        if self.seeker_level == 0: # Naive: 50/50 scan/exploit
            p = 0.5
        elif self.exploit_target == -1: # No target known
            p = 1.0 # Must scan
        else:
            p = 0.2 # 20% scan, 80% exploit

        if np.random.rand() < p:
            self.scan(env_state)
            return 'scan'
        else:
            result = self.exploit(env_state)
            if result == 'scan': # Exploit failed and no other target found
                self.scan(env_state)
                return 'scan'
            else:
                # 'success' or 'fail_rescan' (which is handled by next loop)
                return 'exploit'