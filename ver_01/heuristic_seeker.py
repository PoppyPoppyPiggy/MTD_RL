# File: MTD_RL/ver_01/heuristic_seeker.py
import numpy as np
import random
from cti_bridge import CTIBridge # CTI 브리지 임포트
import config

class HeuristicSeeker:
    """
    MTD 상태와 CTI 정보를 기반으로 휴리스틱하게 공격을 선택하는 클래스.
    MTD 전용 학습 루프 (train_mtd_only.py)에서 사용됩니다.
    """
    def __init__(self, cti_bridge: CTIBridge, action_dim: int):
        self.cti = cti_bridge
        self.action_dim = action_dim
        print("HeuristicSeeker initialized.")

    def select_action(self, state: np.ndarray) -> int:
        """
        주어진 상태(state)를 기반으로 최적의 공격(action)을 선택합니다.

        Args:
            state (np.ndarray): 현재 환경 상태.
                                state의 마지막 'action_dim' 개가 MTD 설정으로 가정합니다.

        Returns:
            int: 선택된 공격 (0부터 action_dim - 1 사이의 정수).
        """
        
        # 1. CTI 정보에서 공격 우선순위 가져오기
        # cti_bridge에 get_cti_priorities() 메소드가 있다고 가정합니다.
        try:
            # cti_bridge.py에 get_cti_priorities()가 구현되어 있어야 함
            cti_priorities = self.cti.get_cti_priorities() # 예: [0.1, 0.5, 0.3, 0.8]
            if cti_priorities is None or len(cti_priorities) != self.action_dim:
                # CTI 정보가 없거나 차원이 맞지 않으면 랜덤 우선순위 생성
                # print("Warning: CTI data invalid. Using random priorities.")
                cti_priorities = np.random.rand(self.action_dim)
        except AttributeError:
            # cti_bridge.py에 get_cti_priorities()가 없는 경우
            # print("Warning: get_cti_priorities() not found in CTIBridge. Using random priorities.")
            cti_priorities = np.random.rand(self.action_dim)

        # 2. 현재 MTD 상태 파악
        # 상태 벡터의 마지막 N개가 MTD 설정이라고 가정합니다.
        # (environment.py의 get_state() 로직과 일치해야 함)
        current_mtd_config = state[-self.action_dim:] # 예: [0, 1, 0, 0] (1: 활성, 0: 비활성)

        # 3. 휴리스틱 로직:
        # - MTD가 활성화되지 *않은* 공격 벡터를 찾습니다.
        # - 그 중에서 CTI 우선순위가 가장 높은 공격을 선택합니다.
        
        unmitigated_attacks = []
        for i in range(self.action_dim):
            if current_mtd_config[i] == 0: # MTD 비활성 상태
                unmitigated_attacks.append((i, cti_priorities[i])) # (공격 인덱스, CTI 우선순위)

        if unmitigated_attacks:
            # CTI 우선순위를 기준으로 내림차순 정렬
            unmitigated_attacks.sort(key=lambda x: x[1], reverse=True)
            # 우선순위가 가장 높은 공격 선택
            best_attack = unmitigated_attacks[0][0]
            # print(f"Heuristic choice: Unmitigated attack {best_attack} (CTI Prio: {unmitigated_attacks[0][1]})")
            return best_attack
        else:
            # 4. 모든 공격이 MTD에 의해 방어되고 있는 경우:
            # - CTI 우선순위가 가장 높은 공격을 선택 (MTD를 우회하려는 시도)
            best_attack = np.argmax(cti_priorities)
            # print(f"Heuristic choice: All mitigated. Attacking highest CTI prio {best_attack}")
            return best_attack