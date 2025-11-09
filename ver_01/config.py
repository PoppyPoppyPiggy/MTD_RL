import argparse
import torch
import numpy as np

# --- [핵심] 테스트베드 인터페이스 정의 ---
# Config_v21의 MTD_META_ACTIONS (0~6)을 기반으로 Action Space 고정
# 0: ip_cd(1.2), 1: ip_cd(0.8), 2: decoy(1.2), 3: decoy(0.8), 4: bl(1.0), 5: bl(-1.0), 6: none
TESTBED_ACTION_DIM = 7

# Config_v21의 DYN_PARAMS 및 위협 정보를 기반으로 State Space 고정
# [current_ip_cd, current_decoy_ratio, current_bl_level, 
#  scan_alert, exploit_alert, breach_alert, 
#  attacker_knowledge, mtd_cost_rate]
TESTBED_OBS_DIM = 8
# ------------------------------------

def get_config():
    parser = argparse.ArgumentParser(description='MTD-RL (Testbed-Compatible Trainer)')
    
    # --- Environment (Config_v21 DYN_PARAMS) ---
    # 파라미터 범위 및 기본값 정의
    parser.add_argument('--base_ip_cd', type=float, default=30.0)
    parser.add_argument('--min_ip_cd', type=float, default=5.0)
    parser.add_argument('--max_ip_cd', type=float, default=60.0)
    
    parser.add_argument('--base_decoy_ratio', type=float, default=0.05)
    parser.add_argument('--min_decoy_ratio', type=float, default=0.0)
    parser.add_argument('--max_decoy_ratio', type=float, default=0.50)
    
    parser.add_argument('--base_bl_level', type=float, default=1.0)
    parser.add_argument('--min_bl_level', type=float, default=0.0)
    parser.add_argument('--max_bl_level', type=float, default=5.0)

    # --- Attacker (Seeker) ---
    # --- [수정] Seeker 레벨 5단계로 확장 ---
    parser.add_argument('--seeker_level', type=str, default='L2', 
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'], 
                        help='Heuristic seeker behavior level (L0:Stealthy, L1:Low, L2:Moderate, L3:Aggressive, L4:Scanner)')
    
    # --- PPO Hyperparameters (from Config_v21) ---
    parser.add_argument('--max_episodes', type=int, default=50000, help='number of episodes')
    parser.add_argument('--max_timesteps', type=int, default=256, help='max timesteps in one episode (ROLLOUT_STEPS)')
    parser.add_argument('--update_timestep', type=int, default=2048, help='update policy every n timesteps (e.g., N_ENVS * ROLLOUT_STEPS / (single_env))')
    
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--K_epochs', type=int, default=10, help='update policy for K epochs')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    parser.add_argument('--entropy_coef', type=float, default=0.03, help='entropy coefficient')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # --- Rewards & Penalties (from Config_v21) ---
    parser.add_argument('--cost_weight', type=float, default=0.25)
    parser.add_argument('--rew_mtd_block_exploit', type=float, default=1.0)
    parser.add_argument('--rew_mtd_decoy', type=float, default=1.5)
    parser.add_argument('--rew_mtd_block_breach', type=float, default=2.0)
    parser.add_argument('--penalty_mtd_exploit', type=float, default=-3.0)
    parser.add_argument('--penalty_mtd_breach', type=float, default=-5.0)
    parser.add_argument('--penalty_mtd_knowledge_leak', type=float, default=-0.5)
    
    # --- Costs (from Config_v21) ---
    parser.add_argument('--cost_mtd_action', type=float, default=0.05)
    parser.add_argument('--cost_shuffle', type=float, default=0.05)
    parser.add_argument('--cost_decoy_ratio', type=float, default=0.15)
    parser.add_argument('--cost_bl_level', type=float, default=0.25)
    
    # --- Logging ---
    parser.add_argument('--wandb', action='store_true', help='use wandb')
    parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='do not use wandb')
    parser.set_defaults(wandb=True)
    
    parser.add_argument('--project_name', type=str, default='MTD-RL-Testbed-Trainer', help='project name for wandb')
    parser.add_argument('--print_interval', type=int, default=20, help='print interval')
    
    parser.add_argument('--log_dir', type=str, default='./logs', help='directory to save logs')
    parser.add_argument('--save_dir', type=str, default='./models', help='directory to save models')
    # (PPO 모델 저장 파일명. 테스트베드에서 로드할 파일명과 일치시킴)
    parser.add_argument('--policy_name', type=str, default='defender_policy.pth', help='name of the saved policy file')


    args = parser.parse_args()
    return args

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')