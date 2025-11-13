# ver_03/seeker.py
# 'rl/ver_02'의 seeker.py와 동일. (ver_03용으로 복사)
# ver_03의 environment.py가 seeker의 행동을 시뮬레이션하므로,
# 이 파일은 직접 사용되지는 않지만 참조용 또는 향후 연동을 위해 유지.

import numpy as np

class Seeker:
    def __init__(self, n_targets):
        self.n_targets = n_targets
        self.knowledge = np.zeros(n_targets)  # 0: unknown, 1: known
        self.exploit_target = -1

    def scan(self, targets):
        """
        Simulates scanning targets to find active ones.
        'targets' is a list/array where 1 means active, 0 means inactive.
        """
        discovered = []
        for i in range(self.n_targets):
            if targets[i] == 1 and self.knowledge[i] == 0:
                self.knowledge[i] = 1
                discovered.append(i)
                if self.exploit_target == -1:
                    self.exploit_target = i
        return discovered

    def exploit(self, targets):
        """
        Tries to exploit a known target.
        If the target moved (targets[self.exploit_target] == 0),
        the exploit fails, and the seeker must rescan.
        """
        if self.exploit_target == -1:
            # No target known, must scan
            return 'scan'

        if targets[self.exploit_target] == 1:
            # Exploit successful
            return 'success'
        else:
            # Exploit failed, target moved
            self.knowledge[self.exploit_target] = 0 # Mark as unknown
            self.exploit_target = -1
            
            # Check if other known targets exist
            known_targets = np.where(self.knowledge == 1)[0]
            if len(known_targets) > 0:
                self.exploit_target = np.random.choice(known_targets)
                return 'fail_switch' # Switched to another known target
            else:
                return 'fail_rescan' # No other known targets, must rescan

    def reset(self):
        self.knowledge = np.zeros(self.n_targets)
        self.exploit_target = -1

    def choose_action(self, targets_state):
        """
        Simple heuristic:
        1. If exploiting and target is still valid, keep exploiting.
        2. If exploiting and target moved, try to find a new target.
        3. If no target, scan.
        """
        if self.exploit_target != -1:
            action_result = self.exploit(targets_state)
            if action_result == 'success':
                return 'exploit'
            else:
                # 'fail_switch' or 'fail_rescan'
                # If 'fail_switch' happened, exploit_target is updated,
                # so next action can be 'exploit'
                if self.exploit_target != -1:
                    return 'exploit' # Try exploiting new known target
                else:
                    return 'scan' # Must rescan
        else:
            self.scan(targets_state)
            return 'scan'