#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTD-only PPO Trainer (v2)
- MTD_RL/ver_02
- Trains MTD policy against a fixed (or heuristic) Seeker.
- [MODIFIED] Added mtd_scoring.py logic to W&B logging.
"""

import os
import time
from datetime import datetime
import torch
import numpy as np
import wandb
from config import Config
from environment import MTDEnv
from ppo import PPO
from utils import set_seed

def train_mtd_only():
    print("Starting MTD-only PPO Training (v2)")

    # --- 1. Load Configuration ---
    config = Config()
    env_config = config.ENV
    train_config = config.TRAIN
    ppo_config = config.PPO
    
    # --- [MODIFIED] Load Scoring Weights (from mtd_scoring.yaml logic) ---
    W_S_D = 0.5  # w_deception_success
    W_R_A = 0.3  # w_attack_resilience
    W_C_M = 0.2  # w_mtd_cost
    # --- End Modification ---

    set_seed(train_config['seed'])
    
    # --- 2. Initialize Environment ---
    env = MTDEnv(config)
    
    state_dim = env.mtd_state_dim
    action_dim = env.mtd_action_dim
    
    # --- 3. Initialize PPO Agent ---
    device = torch.device(train_config['device'])
    ppo_agent = PPO(
        state_dim,
        action_dim,
        ppo_config['lr'],
        ppo_config['gamma'],
        ppo_config['K_epochs'],
        ppo_config['eps_clip'],
        ppo_config['entropy_coef'],
        device
    )
    
    # --- 4. W&B Initialization ---
    run_name = f"mtd_only_train_{datetime.now().strftime('%Y%m%d_%H%M')}"
    try:
        wandb_run = wandb.init(
            project=train_config['wandb_project'],
            group=train_config['wandb_group'],
            name=run_name,
            config={
                "train_config": train_config,
                "ppo_config": ppo_config,
                "env_config": env_config
            }
        )
        print(f"W&B Run started: {run_name}")
    except Exception as e:
        print(f"Failed to initialize W&B: {e}")
        wandb_run = None

    # --- 5. Training Loop ---
    print(f"Starting training loop... Device: {device}")
    
    # Training parameters
    max_ep_len = train_config['max_ep_len']
    max_training_timesteps = train_config['max_training_timesteps']
    log_freq = train_config['log_freq']
    save_model_freq = train_config['save_model_freq']
    
    # PPO update parameters
    update_timestep = ppo_config['update_timestep']
    
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

    while time_step < max_training_timesteps:
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
            avg_ep_loss_p = avg_ep_loss_p / (time_step / update_timestep) # Avg per update
            avg_ep_loss_v = avg_ep_loss_v / (time_step / update_timestep)
            avg_ep_entropy = avg_ep_entropy / (time_step / update_timestep)
            
            # --- [MODIFIED] Calculate averages for scoring metrics ---
            avg_S_MTD = log_avg_S_MTD / log_freq
            avg_S_D = log_avg_S_D / log_freq
            avg_R_A = log_avg_R_A / log_freq
            avg_C_M = log_avg_C_M / log_freq
            # Totals (N_R, N_A, T_A, T_D) are logged as totals for the period
            # --- End Modification ---

            # Log to W&B
            if wandb_run:
                log_data = {
                    "General/Episode": i_episode,
                    "General/Timestep": time_step,
                    "General/Duration_min": (time.time() - start_time) / 60,
                    "Reward/MTD_Reward": avg_ep_reward,
                    "Loss/Policy_Loss": avg_ep_loss_p,
                    "Loss/Value_Loss": avg_ep_loss_v,
                    "Loss/Entropy": avg_ep_entropy,
                    
                    # --- [MODIFIED] Add MTD Scoring Metrics ---
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
            save_path = os.path.join(
                train_config['model_save_dir'],
                train_config['wandb_group'],
                f"ppo_mtd_only_E{i_episode}_T{time_step}.pth"
            )
            ppo_agent.save(save_path)

    env.close() # Ensure env handles cleanup if needed
    if wandb_run:
        wandb_run.finish()
    print("Training finished.")

if __name__ == '__main__':
    train_mtd_only()