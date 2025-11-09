#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for MTD vs Seeker ARL Framework
- (v22) Testbed-Compatible Version (argparse-based)
- All Config_v21 parameters are exposed as command-line arguments.
- wandb is disabled by default (use --wandb to enable).
"""

import argparse
import torch
import numpy as np

# --- [중요] 테스트베드 호환성 인터페이스 ---
# MTD_full_testbed의 State/Action과 동일하게 고정
# State: (ip_cd, decoy_ratio, bl_level, scan_alert, exploit_alert, breach_alert, knowledge, cost_rate)
TESTBED_OBS_DIM = 8 
# Action: (ip_cd_up, ip_cd_down, decoy_up, decoy_down, bl_up, bl_down, none)
TESTBED_ACTION_DIM = 7 
# ----------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="MTD RL Trainer (Testbed-Compatible)")

    # --- 학습 파라미터 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    parser.add_argument('--max_episodes', type=int, default=50000, help='Max training episodes')
    parser.add_argument('--max_timesteps', type=int, default=200, help='Max timesteps per episode')

    # --- PPO 하이퍼파라미터 (Config_v21 기준) ---
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

    # --- [v21] 보상/페널티/비용 파라미터 ---
    parser.add_argument('--cost_weight', type=float, default=0.25, help='Weight of MTD cost in reward')
    # 1. 방어 보상
    parser.add_argument('--rew_mtd_decoy', type=float, default=1.5, help='Reward for decoying seeker')
    parser.add_argument('--rew_mtd_block_exploit', type=float, default=1.0, help='Reward for blocking exploit')
    parser.add_argument('--rew_mtd_block_breach', type=float, default=2.0, help='Reward for blocking breach')
    # 2. 피격 페널티
    parser.add_argument('--penalty_mtd_knowledge_leak', type=float, default=-0.5, help='Penalty for seeker scan success')
    parser.add_argument('--penalty_mtd_exploit', type=float, default=-3.0, help='Penalty for seeker exploit success')
    parser.add_argument('--penalty_mtd_breach', type=float, default=-5.0, help='Penalty for seeker breach success')
    # 3. MTD 비용
    parser.add_argument('--cost_mtd_action', type=float, default=0.05, help='Base cost for any MTD action')
    parser.add_argument('--cost_shuffle', type=float, default=0.05, help='Cost multiplier for shuffle (ip_cd)')
    parser.add_argument('--cost_decoy_ratio', type=float, default=0.15, help='Cost multiplier for decoy ratio')
    parser.add_argument('--cost_bl_level', type=float, default=0.25, help='Cost multiplier for block level')

    # --- 로깅 및 저장 ---
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory for saved models')
    parser.add_argument('--policy_name', type=str, default='mtd_policy.pth', help='Name of the saved policy file')
    parser.add_argument('--print_interval', type=int, default=20, help='Interval to print avg reward')
    parser.add_argument('--save_interval', type=int, default=1000, help='Interval to save model')

    # --- [수정] wandb 로깅 (기본 비활성화) ---
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='MTD-RL-Testbed-Trainer', help='wandb project name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Test: 스크립트가 잘 실행되는지 확인
    args = parser.parse_args()
    print("--- Configuration ---")
    print(f"Seeker Level: {args.seeker_level}")
    print(f"Device: {args.device}")
    print(f"Wandb Logging: {args.wandb}")
    print(f"Testbed Obs Dim: {TESTBED_OBS_DIM}")
    print(f"Testbed Act Dim: {TESTBED_ACTION_DIM}")
    print("---------------------")