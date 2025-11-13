# ver_03/config.py
# 'rl/ver_02'의 config.py에서 'rl/ver_03' 로드맵에 맞게 대폭 수정됨

import torch
import numpy as np

# --- General ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
WANDB_PROJECT = "mtd_rl_ver_03"

# --- MTD Environment (MTDEnv) Parameters ---

# 1. State Space (S) Configuration (로드맵 2.2절)
# S_vector = S_cumulative | S_threat | S_config
# S_cumulative (3 dims)
CUMULATIVE_METRICS_DIM = 3  # [S_D_cumulative, R_A_cumulative, C_M_cumulative]
# S_threat (4 dims)
THREAT_STATE_DIM = 4        # [total_scans_recent, real_target_scans, decoy_scans, new_attacker_flag]
# S_config (4 dims)
MTD_CONFIG_DIM = 4          # [current_strategy_id, param1, param2, time_since_last_mtd]

# Total State Dimension
STATE_DIM = CUMULATIVE_METRICS_DIM + THREAT_STATE_DIM + MTD_CONFIG_DIM # 3 + 4 + 4 = 11

# 2. Action Space (A) Configuration (PAMDP) (로드맵 2.1절)
# A = (a_discrete, a_continuous)
DISCRETE_ACTION_DIM = 4     # 0: IP_Shuffle, 1: Port_Hopping, 2: Decoy_Activation, 3: No_Op
CONTINUOUS_ACTION_DIM = 2   # [param1, param2] (e.g., strength, radius). 
                            # 각 이산 행동에 필요한 파라미터만 사용 (나머지는 무시)

# 3. Reward Function (R) Weights (로드맵 2.3절)
# R_t = (w_d * delta_S_D) + (w_a * delta_R_A) - (w_m * delta_C_M)
REWARD_WEIGHTS = {
    'w_d': 1.0,  # 기만 성공 (S_D) 가중치
    'w_a': 1.0,  # 자원 가용성 (R_A) 가중치
    'w_m': 0.2   # MTD 비용 (C_M) 가중치 (비용이므로 음수 보상)
}

# --- PPO Algorithm Parameters ---
# PPO 하이퍼파라미터 (ver_02와 유사하게 유지)
class PPOConfig:
    def __init__(self):
        self.mode = "train"
        self.gamma = 0.99
        self.lr_actor = 3e-4
        self.lr_critic = 1e-3
        self.k_epochs = 40
        self.eps_clip = 0.2
        self.v_clip = 0.2
        self.T_horizon = 2048
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        
        # ver_03에 맞게 state/action dim 설정
        self.state_dim = STATE_DIM
        self.discrete_action_dim = DISCRETE_ACTION_DIM
        self.continuous_action_dim = CONTINUOUS_ACTION_DIM

# --- Training Parameters ---
MAX_TIMESTEPS = int(3e6)      # 총 학습 타임스텝
SAVE_MODEL_FREQ = int(1e4)    # 모델 저장 주기
LOG_DATA_FREQ = int(2e3)      # 로그 기록 주기 (T_horizon과 일치시키는 것이 좋음)
TARGET_AVG_REWARD = 300       # 학습 종료 목표 보상 (조정 필요)

# --- Simulation Parameters (for Environment) ---
MAX_EPISODE_STEPS = 500       # 에피소드 최대 길이
ATTACKER_SCAN_INTERVAL = 5    # 공격자가 스캔을 시도하는 평균 간격 (steps)
SERVICE_DOWNTIME_PENALTY = -1.0 # MTD로 인한 서비스 다운타임 발생 시 R_A 페널티
MTD_ACTION_COST = 0.1         # MTD 행동 1회 수행 시 기본 C_M 비용