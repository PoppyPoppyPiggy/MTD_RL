# ver_03/environment.py
# 'rl/ver_02'의 environment.py를 'rl/ver_03' 로드맵에 맞게 전면 재설계
# - 'Scorer' 로직 통합 (로드맵 1단계)
# - 새로운 상태(S), 행동(A), 보상(R) 구현 (로드맵 3단계)

import gym
from gym import spaces
import numpy as np
import time
import random
import logging

# ver_03 config 임포트
from ver_03.config import (
    STATE_DIM, DISCRETE_ACTION_DIM, CONTINUOUS_ACTION_DIM, 
    REWARD_WEIGHTS, MAX_EPISODE_STEPS, DEVICE,
    ATTACKER_SCAN_INTERVAL, SERVICE_DOWNTIME_PENALTY, MTD_ACTION_COST
)

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Scorer:
    """
    'mtd_scoring.py'의 로직을 대체 및 통합. (로드맵 1단계)
    MTD 성능 지표 (S_D, R_A, C_M)를 누적 관리.
    """
    def __init__(self, reward_weights):
        self.reward_weights = reward_weights
        self.reset()

    def reset(self):
        # 누적 지표
        self.s_d_cumulative = 0.0
        self.r_a_cumulative = 1.0  # 1.0에서 시작 (100% 가용성)
        self.c_m_cumulative = 0.0
        
        # 이전 step의 지표 (델타 계산용)
        self.prev_s_d = 0.0
        self.prev_r_a = 1.0
        self.prev_c_m = 0.0
        
        logger.info("[Scorer] Reset cumulative metrics.")

    def update_metrics(self, decoy_scanned, service_down, mtd_cost):
        """
        ThreatMonitor와 MTDExecutor로부터 이벤트를 받아 지표를 업데이트.
        """
        # S_D (기만) 업데이트
        if decoy_scanned:
            self.s_d_cumulative += 1.0
            
        # R_A (가용성) 업데이트
        if service_down:
            self.r_a_cumulative += SERVICE_DOWNTIME_PENALTY # 페널티 적용
            
        # C_M (비용) 업데이트
        self.c_m_cumulative += mtd_cost

    def calculate_reward(self):
        """
        '델타 보상 함수' 계산 (로드맵 2.3절)
        R_t = (w_d * delta_S_D) + (w_a * delta_R_A) - (w_m * delta_C_M)
        """
        delta_s_d = self.s_d_cumulative - self.prev_s_d
        delta_r_a = self.r_a_cumulative - self.prev_r_a
        delta_c_m = self.c_m_cumulative - self.prev_c_m

        # 델타 보상 계산
        reward = (self.reward_weights['w_d'] * delta_s_d) + \
                 (self.reward_weights['w_a'] * delta_r_a) - \
                 (self.reward_weights['w_m'] * delta_c_m)
        
        # 다음 계산을 위해 현재 값을 prev로 저장
        self.prev_s_d = self.s_d_cumulative
        self.prev_r_a = self.r_a_cumulative
        self.prev_c_m = self.c_m_cumulative

        return reward, delta_s_d, delta_r_a, delta_c_m

    def get_cumulative_metrics(self):
        return [self.s_d_cumulative, self.r_a_cumulative, self.c_m_cumulative]


class ThreatMonitor:
    """
    'Seeker'의 행동(스캔)을 시뮬레이션하고 '공격 스캔 히트맵' 데이터를 생성.
    (로드맵 2단계 - 'Seeker 스캔 로깅' 및 '로그 파서'의 시뮬레이션 버전)
    """
    def __init__(self):
        self.real_ip = "10.0.0.10"
        self.decoy_ips = ["10.0.0.100", "10.0.0.101"]
        self.current_target_ip = self.real_ip # 공격자의 현재 타겟
        self.reset()

    def reset(self):
        self.total_scans_recent = 0
        self.real_target_scans = 0
        self.decoy_scans = 0
        self.new_attacker_flag = 0.0 # 0 또는 1
        
        # MTD에 의해 변경될 수 있는 시스템 설정
        self.active_decoys = []
        self.ip_shuffled = False

    def simulate_seeker_scan(self):
        """Seeker의 스캔 행동 시뮬레이션"""
        decoy_scanned_event = False
        
        # 일정 확률로 스캔 발생
        if random.random() < (1.0 / ATTACKER_SCAN_INTERVAL):
            self.total_scans_recent += 1
            
            # MTD가 IP를 변경했다면(ip_shuffled), 시커는 일정 확률로 타겟을 잃음
            if self.ip_shuffled:
                if random.random() < 0.5: # 50% 확률로 타겟 재탐색
                    self.current_target_ip = self.real_ip
                    logger.debug("[ThreatMonitor] Seeker re-discovered real IP.")
                else:
                    # 타겟을 잃고 활성 디코이 중 하나를 스캔
                    if self.active_decoys:
                        self.current_target_ip = random.choice(self.active_decoys)
                        logger.debug(f"[ThreatMonitor] Seeker lost target, scanning decoy {self.current_target_ip}")
                self.ip_shuffled = False # IP 셔플 효과 1회성으로 처리

            # 스캔 수행
            if self.current_target_ip == self.real_ip:
                self.real_target_scans += 1
            elif self.current_target_ip in self.active_decoys:
                self.decoy_scans += 1
                decoy_scanned_event = True # 기만 성공! (S_D 업데이트용)
                
            # 일정 확률로 새로운 공격자 등장
            if random.random() < 0.01:
                self.new_attacker_flag = 1.0
        else:
            # 스캔이 없으면 최근 위협은 감소
            self.total_scans_recent = max(0, self.total_scans_recent - 0.1)
            self.new_attacker_flag = 0.0

        return decoy_scanned_event

    def update_mtd_config(self, strategy_id, params):
        """MTD Executor로부터 현재 MTD 설정을 받아와서 시뮬레이션에 반영"""
        if strategy_id == 0: # IP Shuffle
            self.ip_shuffled = True # 시커가 타겟을 잃을 수 있음
        elif strategy_id == 2: # Decoy Activation
            strength = params[0] # 파라미터 1: 활성화 강도 (0~1)
            if strength > 0.5 and self.decoy_ips[0] not in self.active_decoys:
                self.active_decoys.append(self.decoy_ips[0])
            elif strength <= 0.5 and self.active_decoys:
                self.active_decoys.pop()
        # (Port Hopping 등 다른 전략에 대한 시뮬레이션 로직 추가 가능)

    def get_threat_state(self):
        # S_threat 벡터 반환 (로드맵 2.2절)
        return [
            self.total_scans_recent,
            self.real_target_scans,
            self.decoy_scans,
            self.new_attacker_flag
        ]


class MTDExecutor:
    """
    'iptables_mtd_controller.py'의 역할을 시뮬레이션. (로드맵 3단계)
    PAMDP 행동을 받아 MTD 실행을 시뮬레이션하고, 그 결과를 반환.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_strategy_id = 3 # 3: No_Op
        self.current_params = [0.0, 0.0]
        self.time_since_last_mtd = 0

    def execute_action(self, discrete_action, continuous_action):
        """
        PAMDP 행동을 실행하고, 비용(mtd_cost)과 가용성(service_down) 이벤트를 반환.
        """
        self.time_since_last_mtd += 1
        mtd_cost = 0.0
        service_down = False

        if discrete_action == 3: # No_Op
            self.current_strategy_id = 3
            self.current_params = [0.0, 0.0]
            return mtd_cost, service_down # 비용 없음

        # MTD 실행
        self.current_strategy_id = discrete_action
        self.current_params = [float(p) for p in continuous_action] # np.array를 float 리스트로
        self.time_since_last_mtd = 0
        
        # MTD 기본 비용
        mtd_cost += MTD_ACTION_COST

        # 행동 파라미터에 따른 추가 비용 및 다운타임 시뮬레이션
        param1 = self.current_params[0] # 예: 셔플 강도 또는 호핑 반경 (0~1)
        param2 = self.current_params[1] # 예: 셔플 횟수 또는 디코이 타입 (0~1)

        if discrete_action == 0: # IP Shuffle
            # 강도(param1)가 높을수록 비용과 다운타임 확률 증가
            mtd_cost += MTD_ACTION_COST * param1 * 5 # 강도에 비례한 추가 비용
            if random.random() < (param1 * 0.1): # 강도 1.0일 때 10% 다운타임 확률
                service_down = True
                logger.warning(f"[MTDExecutor] IP Shuffle (Strength: {param1:.2f}) caused service downtime!")

        elif discrete_action == 1: # Port Hopping
            # 반경(param1)이 클수록 비용 증가
            mtd_cost += MTD_ACTION_COST * param1 * 3
            if random.random() < (param1 * 0.05): # 강도 1.0일 때 5% 다운타임 확률
                service_down = True
                logger.warning(f"[MTDExecutor] Port Hopping (Radius: {param1:.2f}) caused service downtime!")
        
        elif discrete_action == 2: # Decoy Activation
            # 타입(param2)이 고수준 상호작용일수록(>0.5) 비용 증가
            mtd_cost += MTD_ACTION_COST * param2 * 2
            # 디코이 활성화는 다운타임 없음

        logger.info(f"[MTDExecutor] Action Executed: ID={self.current_strategy_id}, Params=[{param1:.2f}, {param2:.2f}], Cost={mtd_cost}, Down={service_down}")
        
        # (로드맵 2단계 - MTD 이력 로깅)
        # 실제 시스템에서는 여기서 JSON 로그를 파일이나 DB에 기록해야 함.
        # log_message = {
        #     "timestamp": time.time(),
        #     "mtd_action_id": self.current_strategy_id,
        #     "mtd_params": self.current_params,
        #     "cost_incurred": mtd_cost,
        #     "service_down": service_down
        # }
        # with open("mtd_execution.log", "a") as f:
        #     f.write(json.dumps(log_message) + "\n")

        return mtd_cost, service_down

    def get_mtd_config_state(self):
        # S_config 벡터 반환 (로드맵 2.2절)
        return [
            float(self.current_strategy_id),
            self.current_params[0],
            self.current_params[1],
            float(self.time_since_last_mtd)
        ]


class MTDEnv(gym.Env):
    """
    로드맵에 따라 재설계된 rl/ver_03 MTD 강화학습 환경
    """
    def __init__(self, seeker_policy=None):
        super(MTDEnv, self).__init__()
        
        # 1. Scorer, Monitor, Executor 통합 (로드맵 1단계)
        self.scorer = Scorer(REWARD_WEIGHTS)
        self.threat_monitor = ThreatMonitor()
        self.mtd_executor = MTDExecutor()
        
        # (시커 로직은 단순 시뮬레이션으로 대체됨)
        # self.seeker_policy = seeker_policy 

        # 2. Action Space (A) 정의 (PAMDP) (로드맵 2.1절)
        # Box(low, high, (dim,))
        self.action_space = spaces.Dict({
            "discrete": spaces.Discrete(DISCRETE_ACTION_DIM),
            "continuous": spaces.Box(low=0.0, high=1.0, shape=(CONTINUOUS_ACTION_DIM,), dtype=np.float32)
            # 참고: PPO 구현이 Dict 공간을 지원하지 않는 경우,
            # 이를 Discrete(4) + Box(2) = 5차원의 Box로 근사해야 할 수 있음.
            # 여기서는 PPO가 Dict를 처리한다고 가정하고 설계. (ppo.py에서 수정 필요)
        })

        # 3. State Space (S) 정의 (로드맵 2.2절)
        # 11차원 Box(low, high, (STATE_DIM,))
        low_state = np.zeros(STATE_DIM, dtype=np.float32)
        high_state = np.full(STATE_DIM, np.inf, dtype=np.float32)
        
        # 일부 상태값의 최대/최소 경계 설정 (정규화를 위해)
        # S_config [strategy_id, param1, param2, time_since_last]
        low_state[7:11] = [0.0, 0.0, 0.0, 0.0]
        high_state[7:11] = [DISCRETE_ACTION_DIM - 1, 1.0, 1.0, MAX_EPISODE_STEPS]
        # S_threat [total_scans, real_scans, decoy_scans, new_attacker_flag]
        low_state[3:7] = [0.0, 0.0, 0.0, 0.0]
        high_state[3:7] = [np.inf, np.inf, np.inf, 1.0]

        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        self.current_step = 0
        logger.info(f"MTDEnv v3 initialized. StateDim: {STATE_DIM}, Action(Discrete): {DISCRETE_ACTION_DIM}, Action(Continuous): {CONTINUOUS_ACTION_DIM}")

    def _get_state(self):
        """
        새로운 상태 벡터 S_vector = S_cumulative | S_threat | S_config 구성
        (로드맵 2.2절)
        """
        s_cumulative = self.scorer.get_cumulative_metrics()     # [S_D, R_A, C_M] (3 dims)
        s_threat = self.threat_monitor.get_threat_state()       # [scans, real, decoy, new_flag] (4 dims)
        s_config = self.mtd_executor.get_mtd_config_state()     # [strategy, p1, p2, time_since] (4 dims)
        
        # 11차원 벡터로 결합
        state = np.concatenate([s_cumulative, s_threat, s_config]).astype(np.float32)
        return state

    def reset(self):
        self.current_step = 0
        self.scorer.reset()
        self.threat_monitor.reset()
        self.mtd_executor.reset()
        
        logger.info("--- MTDEnv v3 Reset ---")
        return self._get_state()

    def step(self, action):
        """
        환경 Step 실행 (로드맵 3단계)
        1. 행동 파싱 및 실행
        2. 위협 시뮬레이션
        3. 지표 및 보상 계산
        4. 다음 상태 반환
        """
        if self.current_step >= MAX_EPISODE_STEPS:
            # 에피소드가 이미 종료되었어야 함 (안전 장치)
            return self._get_state(), 0.0, True, {}

        self.current_step += 1
        
        # 1. 행동 파싱 및 실행 (PAMDP)
        discrete_action = action["discrete"]
        continuous_action = action["continuous"]
        
        # MTD 실행 및 결과(비용, 다운타임) 획득
        mtd_cost, service_down = self.mtd_executor.execute_action(discrete_action, continuous_action)
        
        # MTD 실행 결과를 ThreatMonitor에 반영 (시뮬레이션용)
        self.threat_monitor.update_mtd_config(discrete_action, continuous_action)
        
        # 2. 위협 시뮬레이션
        # Seeker가 MTD가 적용된 환경을 스캔
        decoy_scanned = self.threat_monitor.simulate_seeker_scan()
        
        # 3. 지표 및 보상 계산
        # Scorer에 이벤트 전달
        self.scorer.update_metrics(decoy_scanned, service_down, mtd_cost)
        
        # '델타 보상' 계산
        reward, delta_s_d, delta_r_a, delta_c_m = self.scorer.calculate_reward()

        # 4. 다음 상태 반환
        next_state = self._get_state()
        
        # 종료 조건
        done = self.current_step >= MAX_EPISODE_STEPS
        
        # info 딕셔너리 (디버깅 및 로깅용 - 로드맵 4단계)
        info = {
            "s_d_cumulative": self.scorer.s_d_cumulative,
            "r_a_cumulative": self.scorer.r_a_cumulative,
            "c_m_cumulative": self.scorer.c_m_cumulative,
            "delta_s_d": delta_s_d,
            "delta_r_a": delta_r_a,
            "delta_c_m": delta_c_m,
            "mtd_action_id": discrete_action,
            "mtd_param1": continuous_action[0],
            "mtd_param2": continuous_action[1]
        }

        return next_state, reward, done, info

    def render(self, mode='human'):
        pass # 시각화 로직 (로드맵 4단계 - 히트맵 등)