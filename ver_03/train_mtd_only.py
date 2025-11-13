# ver_03/train_mtd_only.py
# 'rl/ver_02'의 train_mtd_only.py를 'rl/ver_03' 로드맵에 맞게 수정
# - ver_03의 Environment, PPO, Config 사용
# - Wandb 로깅 확장 (로드맵 4단계)

import torch
import numpy as np
import time
import os
import pandas as pd
import wandb
import argparse # <-- 인자 처리를 위해 추가

# ver_03 임포트
from environment import MTDEnv
from ppo import PPO
from config import (
    PPOConfig, DEVICE, SEED, WANDB_PROJECT,
    MAX_TIMESTEPS, SAVE_MODEL_FREQ, LOG_DATA_FREQ, TARGET_AVG_REWARD
)

# 시드 고정
torch.manual_seed(SEED)
np.random.seed(SEED)

def train(seeker_level, output_policy_path):
    print(f"--- MTD RL Training (ver_03) ---")
    print(f"Seeker Level: {seeker_level}")
    print(f"Output Policy: {output_policy_path}")
    print(f"Device: {DEVICE}, Seed: {SEED}")
    
    # 1. Wandb 초기화 (로드맵 4단계)
    # 각 레벨별로 wandb run 이름 지정
    wandb.init(project=WANDB_PROJECT, config=PPOConfig().__dict__, name=f"L{seeker_level}_Seeker_v3")
    
    # 2. 환경 및 PPO 에이전트 초기화
    env = MTDEnv(seeker_level=seeker_level) # <-- 시커 레벨 전달
    config = PPOConfig() # PPO 설정 로드
    
    ppo_agent = PPO(
        config.state_dim, 
        config.discrete_action_dim, 
        config.continuous_action_dim
    )
    
    # 모델 저장 경로
    model_dir = os.path.dirname(output_policy_path) # <-- 인자에서 경로 추출
    os.makedirs(model_dir, exist_ok=True)
    
    # 로그 기록용 변수
    log_running_reward = 0
    log_running_episodes = 0
    log_episode_rewards = []
    log_mtd_actions = []
    log_s_d = []
    log_r_a = []
    log_c_m = []
    
    start_time = time.time()
    
    # 3. 학습 루프
    state = env.reset()
    current_ep_reward = 0
    
    for t in range(1, MAX_TIMESTEPS + 1):
        # 3.1 행동 선택 (PAMDP)
        action_env, action_store, log_prob, value = ppo_agent.select_action(state)
        
        # 3.2 환경 스텝 실행
        # action_env = {"discrete": int, "continuous": np.array}
        state, reward, done, info = env.step(action_env)
        
        # 3.3 버퍼에 저장
        # (reward, log_prob, value, state, action_discrete, action_continuous, done)
        ppo_agent.buffer.append((
            reward, 
            log_prob, 
            value,
            state, # state가 아닌 s_t를 저장해야 하지만, PPO 구현상 편의를 위해 s_{t+1} 저장 후 GAE에서 처리
            action_store["discrete"], 
            action_store["continuous"],
            done
        ))
        
        current_ep_reward += reward

        # 3.4 PPO 업데이트 (버퍼가 찼을 때)
        if len(ppo_agent.buffer) >= config.T_horizon:
            ppo_agent.update()
            
        # 3.5 에피소드 종료 처리
        if done:
            state = env.reset()
            
            # 에피소드 로그 기록
            log_episode_rewards.append(current_ep_reward)
            log_mtd_actions.extend(info.get("mtd_action_id", 0)) # 예시 (실제로는 info에서 리스트로 받아야 함)
            log_s_d.append(info.get("s_d_cumulative", 0))
            log_r_a.append(info.get("r_a_cumulative", 0))
            log_c_m.append(info.get("c_m_cumulative", 0))
            
            log_running_reward += current_ep_reward
            log_running_episodes += 1
            current_ep_reward = 0

        # 3.6 로그 기록 및 모델 저장
        if t % LOG_DATA_FREQ == 0:
            avg_reward = log_running_reward / log_running_episodes if log_running_episodes > 0 else 0
            avg_reward = round(avg_reward, 2)
            
            avg_s_d = np.mean(log_s_d) if log_s_d else 0
            avg_r_a = np.mean(log_r_a) if log_r_a else 0
            avg_c_m = np.mean(log_c_m) if log_c_m else 0

            elapsed_time = time.time() - start_time
            
            print(f"Timestep: {t}/{MAX_TIMESTEPS} | Avg Reward: {avg_reward} | S_D: {avg_s_d:.2f} | R_A: {avg_r_a:.2f} | C_M: {avg_c_m:.2f} | Time: {elapsed_time:.2f}s")
            
            # Wandb 로그 (로드맵 4단계)
            wandb.log({
                "avg_episode_reward": avg_reward,
                "total_timesteps": t,
                "avg_s_d_cumulative": avg_s_d,
                "avg_r_a_cumulative": avg_r_a,
                "avg_c_m_cumulative": avg_c_m,
                "learning_rate_actor": config.lr_actor,
                "episodes_completed": log_running_episodes
            })
            
            # 로그 변수 리셋
            log_running_reward = 0
            log_running_episodes = 0
            log_episode_rewards = []
            log_mtd_actions = []
            log_s_d = []
            log_r_a = []
            log_c_m = []

        # 3.7 모델 저장
        if t % SAVE_MODEL_FREQ == 0:
            print(f"--- Saving model at timestep {t} ---")
            # 중간 저장 모델 경로
            intermediate_path = output_policy_path.replace(".pth", f"_step_{t}.pth")
            ppo_agent.save(intermediate_path)

        # 3.8 학습 종료 조건 (TARGET_AVG_REWARD 도달 시)
        if log_running_episodes > 100 and avg_reward > TARGET_AVG_REWARD:
             print(f"--- Target average reward {TARGET_AVG_REWARD} reached! Stopping training. ---")
             ppo_agent.save(output_policy_path) # <-- 최종 경로에 저장
             break
    
    # 3.9 (수정) Max Timestep 도달 시 최종 저장
    if t >= MAX_TIMESTEPS:
        print(f"--- Max timesteps {MAX_TIMESTEPS} reached. Saving final model. ---")
        ppo_agent.save(output_policy_path)

    env.close()
    wandb.finish()
    print(f"--- MTD RL Training (ver_03) Finished for L{seeker_level} ---")

if __name__ == "__main__":
    # 셸 스크립트에서 인자를 받도록 argparse 추가
    parser = argparse.ArgumentParser(description="Train MTD RL agent (ver_03)")
    parser.add_argument(
        "--seeker_level",
        type=int,
        default=0,
        help="Seeker sophistication level (0-4)"
    )
    parser.add_argument(
        "--output_policy",
        type=str,
        default=f"models/ver_03/L0_Seeker_v3/defender_policy_v3_L0.pth",
        help="Path to save the final trained policy"
    )
    args = parser.parse_args()
    
    # 인자를 train 함수로 전달
    train(seeker_level=args.seeker_level, output_policy_path=args.output_policy)