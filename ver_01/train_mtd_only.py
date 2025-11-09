#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MTD RL Trainer (v22)
- Testbed-Compatible (8D State, 7D Action)
- Uses Config_v21 parameters from config.py
- [KPI] Logs detailed performance metrics (KPIs) to wandb
"""

import os
import torch
import numpy as np
import wandb
from config import get_args, TESTBED_OBS_DIM, TESTBED_ACTION_DIM
from environment import NetworkEnv
from heuristic_seeker import HeuristicSeeker
from ppo import PPO
from utils import set_seed, Array2Tensor

def main(args):
    
    # --- 1. 초기 설정 (시드, wandb, 디렉터리) ---
    set_seed(args.seed)
    
    if args.wandb:
        run_name = f"MTD_Trainer_Seeker_{args.seeker_level}"
        wandb.init(project=args.project_name, name=run_name, config=args)
    
    # 저장 경로 생성
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.policy_name)

    # --- 2. 환경 및 에이전트 초기화 ---
    
    # Seeker (스파링 파트너) 생성
    seeker = HeuristicSeeker(args)
    
    # Env (시뮬레이터) 생성
    env = NetworkEnv(args, seeker)
    
    # PPO (MTD 에이전트) 생성
    ppo_agent = PPO(
        state_dim=TESTBED_OBS_DIM,
        action_dim=TESTBED_ACTION_DIM,
        lr=args.lr,
        gamma=args.gamma,
        K_epochs=args.K_epochs,
        eps_clip=args.eps_clip,
        entropy_coef=args.entropy_coef,
        device=args.device
    )

    print(f"Training MTD Agent against Seeker Level: {args.seeker_level}")
    print(f"State Dim: {TESTBED_OBS_DIM}, Action Dim: {TESTBED_ACTION_DIM}, Device: {args.device}")

    # --- 3. 학습 루프 ---
    
    print_running_reward = 0
    print_running_episodes = 0
    
    # [KPI] 상세 지표 수집을 위한 버퍼
    episode_stats_buffer = []
    
    timestep = 0
    
    for i_episode in range(1, args.max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        for t in range(1, args.max_timesteps + 1):
            timestep += 1
            
            # Action (MTD Policy)
            action, log_prob = ppo_agent.select_action(state)
            
            # Perform action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            # ppo.py의 select_action이 state, action, logprob를 버퍼에 저장
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.dones.append(done)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO
            if timestep % args.update_timestep == 0:
                # [KPI] ppo_agent.update()가 이제 loss 값을 반환
                pl, vl, ent = ppo_agent.update()
                # ppo_agent.buffer.clear() # update() 내부에서 clear() 호출
                
                # [KPI] PPO 학습 지표 로깅
                if args.wandb:
                    wandb.log({
                        'train/policy_loss': pl,
                        'train/value_loss': vl,
                        'train/entropy': ent,
                        'train/episode_at_update': i_episode # (참고용) 현재 에피소드 번호
                    })
                
            if done:
                # [KPI] 에피소드 종료 시, info에서 통계 버퍼로 복사
                if 'episode_stats' in info:
                    episode_stats_buffer.append(info['episode_stats'])
                break
            
        print_running_reward += episode_reward
        print_running_episodes += 1
        
        # --- [KPI] 상세 지표 로깅 (print_interval마다) ---
        if i_episode % args.print_interval == 0 and print_running_episodes > 0:
            avg_reward = print_running_reward / print_running_episodes
            
            log_data = {
                'epoch/episode': i_episode,
                'epoch/avg_reward': avg_reward
            }
            
            if episode_stats_buffer:
                # 리스트의 딕셔너리에서 평균을 계산
                avg_stats = {}
                kpi_stats = {}
                policy_stats = {}
                
                keys = episode_stats_buffer[0].keys()
                for k in keys:
                    # 1. Raw Counts (평균값)
                    avg_stats[f"raw_counts/{k}"] = np.mean([s[k] for s in episode_stats_buffer])
                
                # 2. 파생 KPI 계산
                total_breaches = avg_stats["raw_counts/outcome_breach_success"]
                kpi_stats["kpi/breach_stop_rate"] = 1.0 - total_breaches # (1 에피소드 당 1회 breach만 가능)
                
                total_exploits = avg_stats["raw_counts/seeker_exploits"]
                total_exploit_blocks = avg_stats["raw_counts/outcome_exploit_blocked"]
                kpi_stats["kpi/exploit_block_rate"] = total_exploit_blocks / total_exploits if total_exploits > 0 else 0
                
                total_scans = avg_stats["raw_counts/seeker_scans"]
                total_scan_defense = avg_stats["raw_counts/outcome_scan_blocked"] + avg_stats["raw_counts/outcome_scan_hit_decoy"]
                kpi_stats["kpi/scan_defense_rate"] = total_scan_defense / total_scans if total_scans > 0 else 0
                
                # 3. 시간 기반 KPI
                kpi_stats["kpi/avg_time_to_breach"] = avg_stats["raw_counts/time_to_breach"]
                kpi_stats["kpi/avg_time_to_first_alert"] = avg_stats["raw_counts/time_to_first_alert"]

                # 4. 평균 정책 파라미터 (DRS 대안)
                policy_stats["avg_policy/final_ip_cd"] = avg_stats["raw_counts/final_ip_cd"]
                policy_stats["avg_policy/final_decoy_ratio"] = avg_stats["raw_counts/final_decoy_ratio"]
                policy_stats["avg_policy/final_bl_level"] = avg_stats["raw_counts/final_bl_level"]
                
                log_data.update(avg_stats)
                log_data.update(kpi_stats)
                log_data.update(policy_stats)
                episode_stats_buffer.clear() # 버퍼 비우기

            print(f"Episode: {i_episode}, Avg Reward: {avg_reward:.2f}")
            if args.wandb:
                wandb.log(log_data)
            
            print_running_reward = 0
            print_running_episodes = 0
            
        # 모델 저장
        if i_episode % args.save_interval == 0:
            ppo_agent.save(save_path)
            
    env.close()
    if args.wandb:
        wandb.finish()
        
if __name__ == '__main__':
    args = get_args()
    main(args)