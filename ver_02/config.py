#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration File for MTD vs Seeker ARL Framework
- (v23) Testbed-Compatible Version (argparse-based)
- [수정] State/Action 차원을 배포 환경(iptables) 기준으로 통일
- [수정] 보상/비용 파라미터를 배포 환경(DNAT) 기준으로 단순화
- [MODIFIED] --wandb_group 인자 추가 (W&B 호환성)
"""

import argparse
import torch
import numpy as np
import os # [MODIFIED] os 임포트

# --- [중요] 테스트베드/배포 환경 호환성 인터페이스 ---
# mtd_state_reader.py / iptables_mtd.yaml 과 완벽히 일치
# State: (R1, R2, R3, R4, R5, R6, Decoy, Alert)
TESTBED_OBS_DIM = 8
# Action: (DNAT_R1, DNAT_R2, ..., DNAT_R6, DNAT_Decoy)
TESTBED_ACTION_DIM = 7
# ----------------------------------------

# [MODIFIED] Seeker 정책 파일 경로 (ver_02 기준)
# ver_02/L0_Seeker/defender_policy_L0.pth ...
BASE_POLICY_DIR = os.path.join(os.path.dirname(__file__))

def get_seeker_policy_path(level):
    """주어진 레벨에 대한 Seeker 정책 파일 경로를 반환합니다."""
    if level == "L0":
        return None # L0는 Heuristic이므로 정책 파일 없음
    
    policy_path = os.path.join(
        BASE_POLICY_DIR,
        f"{level}_Seeker",
        f"defender_policy_{level}.pth"
    )
    
    if not os.path.exists(policy_path):
        print(f"Warning: Seeker policy file not found at {policy_path}")
        print("         (This is normal for L0, but an error for L1-L4)")
        return None
    return policy_path

def get_args():
    parser = argparse.ArgumentParser(description="MTD RL Trainer (Testbed-Compatible v23)")

    # --- 학습 파라미터 ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda/cpu)')
    parser.add_argument('--max_episodes', type=int, default=500, help='Max training episodes') # 기본값 500으로 수정
    parser.add_argument('--max_timesteps', type=int, default=200, help='Max timesteps per episode')

    # --- PPO 하이퍼파라미터 (v21 기준) ---
    parser.add_argument('--update_timestep', type=int, default=200, help='Timesteps to update policy') # 400 -> 200
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--K_epochs', type=int, default=3, help='PPO update epochs') # 10 -> 3
    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient') # 0.03 -> 0.01
    
    # --- Seeker 설정 (Config_v21 기준 5단계) ---
    parser.add_argument('--seeker_level', type=str, default='L0', 
                        choices=['L0', 'L1', 'L2', 'L3', 'L4'], 
                        help='Seeker behavior level (L0 to L4)')

    # --- [v23 수정] 보상/페널티/비용 파라미터 (DNAT 기반) ---
    # [MODIFIED] ver_02/cti_bridge.py의 보상 체계와 일치시킴
    parser.add_argument('--rew_mtd_decoy', type=float, default=0.5, help='Reward for decoying seeker (scan/exploit)')
    parser.add_argument('--penalty_mtd_breach', type=float, default=-1.0, help='Penalty for seeker breach success on Real Target')
    parser.add_argument('--cost_mtd_action', type=float, default=-0.1, help='Base cost for any MTD action (per step cost)')
    parser.add_argument('--cost_mtd_decoy_action', type=float, default=-0.05, help='Additional cost for decoy MTD action')
    # (참고: cost_shuffle은 ver_02/environment.py에 구현되어 있지 않음)

    # --- 로깅 및 저장 ---
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for logs')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory for saved models')
    parser.add_argument('--policy_name', type=str, default='mtd_policy.pth', help='Name of the saved policy file')
    parser.add_argument('--print_interval', type=int, default=10, help='Interval to print avg reward') # 20 -> 10
    parser.add_argument('--save_interval', type=int, default=50, help='Interval to save model') # 1000 -> 50

    # --- [수정] wandb 로깅 (기본 비활성화) ---
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='mtd_testbed_live', help='wandb project name (mtd_scoring.py와 일치)')
    
    # [MODIFIED] W&B 그룹 인자 추가 (run_all_experiments.sh에서 전달받음)
    parser.add_argument('--wandb_group', type=str, default='vs_L4_Seeker', help='wandb group name (예: vs_L4_Seeker)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Test: 스크립트가 잘 실행되는지 확인
    args = get_args()
    print("--- Configuration (v23 - Testbed Compatible) ---")
    print(f"Seeker Level: {args.seeker_level}")
    print(f"Device: {args.device}")
    print(f"Wandb Logging: {args.wandb}")
    print(f"W&B Group: {args.wandb_group}")
    print(f"Testbed Obs Dim: {TESTBED_OBS_DIM}")
    print(f"Testbed Act Dim: {TESTBED_ACTION_DIM}")
    print(f"Reward (Decoy Hit): {args.rew_mtd_decoy}")
    print(f"Penalty (Breach): {args.penalty_mtd_breach}")
    print(f"Cost (Action): {args.cost_mtd_action}")
    print(f"Seeker Policy Path: {get_seeker_policy_path(args.seeker_level)}")
    print("---------------------------------")