# File: ver_02/config.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for MTD vs Seeker ARL Framework
- (v23) Testbed-Compatible Version (argparse-based)
- [수정] State/Action 차원을 배포 환경(iptables) 기준으로 통일
- [수정] 보상/비용 파라미터를 배포 환경(DNAT) 기준으로 단순화
"""

import argparse
import torch
import numpy as np

# --- [중요] 테스트베드/배포 환경 호환성 인터페이스 ---
# mtd_state_reader.py / iptables_mtd.yaml 과 완벽히 일치
# State: (R1, R2, R3, R4, R5, R6, Decoy, Alert)
TESTBED_OBS_DIM = 8 
# Action: (DNAT_R1, DNAT_R2, ..., DNAT_R6, DNAT_Decoy)
TESTBED_ACTION_DIM = 7 
# ----------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="MTD RL Trainer (Testbed-Compatible v23)")

    # --- 학습 파라미터 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    parser.add_argument('--max_episodes', type=int, default=50000, help='Max training episodes')
    parser.add_argument('--max_timesteps', type=int, default=200, help='Max timesteps per episode')

    # --- PPO 하이퍼파라미터 (v21 기준) ---
    parser.add_argument('--update_timestep', type=int, default=400, help='Timesteps to update policy')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update epochs')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--entropy_coef', type=float, default=0.03, help='Entropy coefficient')
    
    # --- Seeker 설정 (Config_v21 기준 5단계) ---
    parser.add_argument('--seeker_level', type=str, default='L0', 
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'], 
                        help='Seeker behavior level (L0 to L4)')

    # --- [v23 수정] 보상/페널티/비용 파라미터 (DNAT 기반) ---
    parser.add_argument('--cost_weight', type=float, default=0.25, help='Weight of MTD cost in reward')
    
    # 1. 방어 보상 (Seeker를 속이거나 막았을 때)
    parser.add_argument('--rew_mtd_decoy', type=float, default=1.5, help='Reward for decoying seeker (scan/exploit)')
    # (v23에서는 Decoy 보상으로 통일)
    parser.add_argument('--rew_mtd_block_exploit', type=float, default=1.5, help='(v23) Alias for rew_mtd_decoy')
    parser.add_argument('--rew_mtd_block_breach', type=float, default=1.5, help='(v23) Alias for rew_mtd_decoy')
    
    # 2. 피격 페널티 (Seeker가 성공했을 때)
    parser.add_argument('--penalty_mtd_knowledge_leak', type=float, default=-0.5, help='Penalty for seeker scan success on Real Target')
    parser.add_argument('--penalty_mtd_exploit', type=float, default=-5.0, help='(v23) Alias for penalty_mtd_breach')
    parser.add_argument('--penalty_mtd_breach', type=float, default=-5.0, help='Penalty for seeker breach success on Real Target')
    
    # 3. MTD 비용 (행동 자체의 비용)
    parser.add_argument('--cost_mtd_action', type=float, default=0.01, help='Base cost for any MTD action (per step cost)')
    parser.add_argument('--cost_shuffle', type=float, default=0.05, help='[v23] Additional cost if action_id changes')

    # --- 로깅 및 저장 ---
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory for saved models')
    parser.add_argument('--policy_name', type=str, default='mtd_policy.pth', help='Name of the saved policy file')
    parser.add_argument('--print_interval', type=int, default=20, help='Interval to print avg reward')
    parser.add_argument('--save_interval', type=int, default=1000, help='Interval to save model')

    # --- [수정] wandb 로깅 (기본 비활성화) ---
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='MTD-RL-Testbed-Trainer-v23', help='wandb project name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Test: 스크립트가 잘 실행되는지 확인
    args = get_args()
    print("--- Configuration (v23 - Testbed Compatible) ---")
    print(f"Seeker Level: {args.seeker_level}")
    print(f"Device: {args.device}")
    print(f"Wandb Logging: {args.wandb}")
    print(f"Testbed Obs Dim: {TESTBED_OBS_DIM}")
    print(f"Testbed Act Dim: {TESTBED_ACTION_DIM}")
    print(f"Reward (Decoy Hit): {args.rew_mtd_decoy}")
    print(f"Penalty (Breach): {args.penalty_mtd_breach}")
    print(f"Cost (Action): {args.cost_mtd_action}")
    print(f"Cost (Shuffle): {args.cost_shuffle}")
    print("---------------------------------")