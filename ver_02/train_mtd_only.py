#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTD-only PPO Trainer (v2)
- MTD_RL/ver_02
- [MODIFIED] ver_03의 argparse (get_args) 설정 방식을 사용하도록 수정
- [MODIFIED] mtd_scoring.py와 동일한 W&B 로그 키/그룹 사용
"""

import os
import time
from datetime import datetime
import torch
import numpy as np
import wandb
import argparse
# [MODIFIED] Config 클래스 대신 get_args 함수와 TESTBED 차원 임포트
from config import get_args, TESTBED_OBS_DIM, TESTBED_ACTION_DIM
from environment import MTDEnv
from ppo import PPO
from utils import set_seed

# main logic wrapped in a function to accept args
def train_mtd_only(args):
    print(f"Starting MTD-only PPO Training (v2) for Seeker Level: {args.seeker_level}")

    # --- 1. Load Configuration ---
    # args는 main에서 이미 파싱됨
    
    # --- [MODIFIED] Load Scoring Weights (from mtd_scoring.yaml logic) ---
    W_S_D = 0.5  # w_deception_success
    W_R_A = 0.3  # w_attack_resilience
    W_C_M = 0.2  # w_mtd_cost
    # --- End Modification ---

    set_seed(args.seed)
    
    # --- 2. Initialize Environment ---
    # [MODIFIED] MTDEnv에 args 객체 전달
    env = MTDEnv(args) 
    
    state_dim = TESTBED_OBS_DIM   # config.py에서 임포트
    action_dim = TESTBED_ACTION_DIM # config.py에서 임포트
    
    # --- 3. Initialize PPO Agent ---
    device = torch.device(args.device)
    ppo_agent = PPO(
        state_dim,
        action_dim,
        args.lr,
        args.gamma,
        args.K_epochs,
        args.eps_clip,
        args.entropy_coef,
        device
    )
    
    # --- 4. W&B Initialization ---
    run_name = f"mtd_only_train_{args.seeker_level}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # [MODIFIED] Use args for W&B project and group
    if args.wandb:
        try:
            wandb_run = wandb.init(
                project=args.project_name, # config.py의 --project_name
                group=args.wandb_group,   # run_all_experiments.sh에서 전달
                name=run_name,
                config=vars(args) # Log command line arguments
            )
            print(f"W&B Run started: {run_name} (Group: {args.wandb_group})")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            wandb_run = None
    else:
        wandb_run = None

    # --- 5. Training Loop ---
    print(f"Starting training loop... Device: {device}")
    
    # Training parameters from args
    max_ep_len = args.max_timesteps
    log_freq = args.print_interval
    save_model_freq = args.save_interval
    
    # PPO update parameters from args
    update_timestep = args.update_timestep
    
    time_step = 0
    i_episode = 0
    
    # Logging accumulators (reset every log_freq)
    avg_ep_reward = 0
    avg_ep_loss_p = 0
    avg_ep_loss_v = 0
    avg_ep_entropy = 0
    
    # --- [MODIFIED] Scoring accumulators (reset every log_freq) ---
    log_avg_S_MTD = 0
    log_avg_S_D = 0
    log_avg_R_A = 0
    log_avg_C_M = 0
    log_total_N_R = 0
    log_total_N_A = 0
    log_total_T_A = 0  # Total Attack Steps
    log_total_T_D = 0  # Total Decoy Steps (while attacked)
    # --- End Modification ---

    start_time = time.time()

    # [MODIFIED] Loop condition changed from timesteps to episodes
    while i_episode < args.max_episodes:
        i_episode += 1
        state = env.reset()
        ep_reward = 0
        
        # --- [MODIFIED] Episode-level scoring accumulators ---
        ep_total_attack_steps = 0
        ep_steps_on_decoy_while_attacked = 0
        ep_successful_attacks_N_A = 0
        ep_reconfigurations_N_R = 0
        ep_total_cost = 0
        last_mtd_action = -1  # Invalid action to ensure first action counts
        # --- End Modification ---

        for t in range(1, max_ep_len + 1):
            time_step += 1
            
            # Select action
            mtd_action, log_prob = ppo_agent.select_action(state)
            
            # Perform action
            state, reward, done, step_info = env.step(mtd_action)
            
            # Store experience
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.dones.append(done)
            
            ep_reward += reward
            
            # --- [MODIFIED] Accumulate scoring info from this step ---
            ep_total_cost += step_info.get('cost', 0)
            is_attack = step_info.get('is_attack_detected', False)
            is_decoy = step_info.get('is_decoy_action', False)
            is_breach = step_info.get('is_breach', False)
            
            if is_attack:
                ep_total_attack_steps += 1
            if is_attack and is_decoy:
                ep_steps_on_decoy_while_attacked += 1
            if is_breach: # 'is_breach' is already defined as (success AND NOT decoy)
                ep_successful_attacks_N_A += 1
                
            # Count reconfigurations (N_R)
            if mtd_action != last_mtd_action and t > 1:
                ep_reconfigurations_N_R += 1
            last_mtd_action = mtd_action
            # --- End Modification ---

            # PPO Update
            if time_step % update_timestep == 0:
                loss_p, loss_v, entropy = ppo_agent.update()
                avg_ep_loss_p += loss_p
                avg_ep_loss_v += loss_v
                avg_ep_entropy += entropy
                
            if done:
                break
                
        # --- [MODIFIED] Calculate episode-level scores ---
        ep_cost_C_M = ep_total_cost / t  # Avg cost per step
        ep_metric_S_D = (ep_steps_on_decoy_while_attacked / ep_total_attack_steps) if ep_total_attack_steps > 0 else 0.0
        ep_metric_N_A = ep_successful_attacks_N_A
        ep_metric_N_R = ep_reconfigurations_N_R
        
        # R_A = N_R / N_A. If N_A is 0, score is min(N_R, 10.0)
        ep_metric_R_A = min(ep_metric_N_R, 10.0) if ep_metric_N_A == 0 else (ep_metric_N_R / ep_metric_N_A)
        
        # S_MTD = w1*S_D + w2*R_A - w3*C_M
        ep_metric_S_MTD = (W_S_D * ep_metric_S_D) + (W_R_A * ep_metric_R_A) - (W_C_M * ep_cost_C_M)
        
        # Accumulate for logging period
        avg_ep_reward += ep_reward
        log_avg_S_MTD += ep_metric_S_MTD
        log_avg_S_D += ep_metric_S_D
        log_avg_R_A += ep_metric_R_A
        log_avg_C_M += ep_cost_C_M
        log_total_N_R += ep_metric_N_R
        log_total_N_A += ep_metric_N_A
        log_total_T_A += ep_total_attack_steps
        log_total_T_D += ep_steps_on_decoy_while_attacked
        # --- End Modification ---

        # Logging
        if i_episode % log_freq == 0:
            avg_ep_reward = avg_ep_reward / log_freq
            
            # [MODIFIED] PPO 업데이트가 발생한 경우에만 loss 평균 계산
            update_counts = (time_step // update_timestep) - ((time_step - (t * log_freq)) // update_timestep)
            if update_counts > 0:
                avg_ep_loss_p = avg_ep_loss_p / update_counts
                avg_ep_loss_v = avg_ep_loss_v / update_counts
                avg_ep_entropy = avg_ep_entropy / update_counts
            
            # --- [MODIFIED] Calculate averages for scoring metrics ---
            avg_S_MTD = log_avg_S_MTD / log_freq
            avg_S_D = log_avg_S_D / log_freq
            avg_R_A = log_avg_R_A / log_freq
            avg_C_M = log_avg_C_M / log_freq
            # --- End Modification ---

            # Log to W&B
            if wandb_run:
                log_data = {
                    # [MODIFIED] X축을 "General/global_step" (시간)이 아닌 "General/Timestep"으로 설정
                    # mtd_scoring.py와 X축을 통일하려면 wandb.define_metric("General/Timestep") 필요
                    "General/Episode": i_episode,
                    "General/Timestep": time_step,
                    "General/Duration_min": (time.time() - start_time) / 60,
                    "Reward/MTD_Reward": avg_ep_reward,
                    "Loss/Policy_Loss": avg_ep_loss_p,
                    "Loss/Value_Loss": avg_ep_loss_v,
                    "Loss/Entropy": avg_ep_entropy,
                    
                    # --- [MODIFIED] Add MTD Scoring Metrics (Keys identical to mtd_scoring.py) ---
                    "Metric/MTD_Score_Overall": avg_S_MTD,
                    "Metric/Metric_Deception_Success (S_D)": avg_S_D,
                    "Metric/Metric_Attack_Resilience (R_A)": avg_R_A,
                    "Metric/Metric_MTD_Cost (C_M)": avg_C_M,
                    "Metric/Detail_Reconfigurations (N_R)": log_total_N_R,
                    "Metric/Detail_Successful_Attacks (N_A)": log_total_N_A,
                    "Metric/Detail_Total_Attack_Steps (T_A)": log_total_T_A,
                    "Metric/Detail_Total_Decoy_Steps (T_D)": log_total_T_D
                    # --- End Modification ---
                }
                wandb_run.log(log_data)

            print(f"Episode: {i_episode} \t Timestep: {time_step} \t Reward: {avg_ep_reward:.2f} \t S_MTD: {avg_S_MTD:.2f}")

            # Reset logging accumulators
            avg_ep_reward = 0
            avg_ep_loss_p = 0
            avg_ep_loss_v = 0
            avg_ep_entropy = 0
            
            # --- [MODIFIED] Reset scoring accumulators ---
            log_avg_S_MTD = 0
            log_avg_S_D = 0
            log_avg_R_A = 0
            log_avg_C_M = 0
            log_total_N_R = 0
            log_total_N_A = 0
            log_total_T_A = 0
            log_total_T_D = 0
            # --- End Modification ---

        # Save model
        if i_episode % save_model_freq == 0:
            # [MODIFIED] Use args for save path and policy name
            save_path = os.path.join(
                args.save_dir,
                args.policy_name
            )
            ppo_agent.save(save_path)

    # [MODIFIED] Save final model
    final_save_path = os.path.join(args.save_dir, args.policy_name)
    ppo_agent.save(final_save_path)
    print(f"Final model saved at: {final_save_path}")

    # env.close() # MTDEnv에 close()가 정의되어 있지 않으면 주석 처리
    if wandb_run:
        wandb_run.finish()
    print("Training finished.")

# [MODIFIED] Add main block with argparse
if __name__ == '__main__':
    args = get_args() # config.py에서 인자 파싱
    train_mtd_only(args)