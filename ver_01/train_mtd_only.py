import torch
import numpy as np
import wandb
import os # --- 수정: os 모듈 추가 ---
import random

# --- [핵심] 수정된 Config 파일 및 고정 인터페이스 Import ---
from config import get_config, DEVICE, TESTBED_OBS_DIM, TESTBED_ACTION_DIM
from environment import NetworkEnv # <-- 'NetworkEnvironment'가 아니라 'NetworkEnv'입니다.
from ppo import PPO
from utils import Array2Tensor, set_seed # --- [수정] set_seed Import ---
from heuristic_seeker import HeuristicSeeker # 수정된 Seeker Import

def main(args):
    
    # --- [수정] set_seed 함수 사용 ---
    set_seed(args.seed)

    # Environment
    env = NetworkEnv(args) # <-- 'NetworkEnvironment'가 아니라 'NetworkEnv'입니다.
    
    # Seeker (Attacker)
    seeker = HeuristicSeeker(args)
    env.set_seeker(seeker)
    
    # PPO Agent
    # --- [핵심] 고정된 obs_dim, act_dim 사용 ---
    ppo_agent = PPO(
        state_dim=TESTBED_OBS_DIM,    # <-- [수정] 'obs_dim' -> 'state_dim'
        action_dim=TESTBED_ACTION_DIM, # <-- [수정] 'act_dim' -> 'action_dim'
        lr=args.lr,
        gamma=args.gamma,
        K_epochs=args.K_epochs,
        eps_clip=args.eps_clip,
        device=DEVICE,
        entropy_coef=args.entropy_coef # 엔트로피 계수 추가
    )
    
    # Logging
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        wandb.init(
            config=vars(args),
            project=args.project_name,
            name=f'MTD_Trainer_Seeker_{args.seeker_level}',
            dir=args.log_dir
        )

    # Training
    print_running_reward = 0
    print_running_episodes = 0
    
    timestep = 0
    
    for i_episode in range(1, args.max_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        for t in range(args.max_timesteps):
            timestep += 1
            
            # Select action
            action, log_prob = ppo_agent.select_action(state)
            
            # Perform action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ppo_agent.buffer.states.append(Array2Tensor(state, DEVICE))
            ppo_agent.buffer.actions.append(action)
            ppo_agent.buffer.logprobs.append(log_prob)
            
            state = next_state
            episode_reward += reward
            
            # Update PPO
            if timestep % args.update_timestep == 0:
                ppo_agent.update()
                ppo_agent.buffer.clear()
                # timestep = 0 # (주석 처리: Timestep은 계속 누적되어야 할 수 있음, 정책에 따라 다름)
                
            if done:
                break
            
        print_running_reward += episode_reward
        print_running_episodes += 1
        
        # Logging
        if i_episode % args.print_interval == 0 and print_running_episodes > 0:
            avg_reward = print_running_reward / print_running_episodes
            print(f"Episode: {i_episode}, Avg Reward: {avg_reward:.2f}")
            if args.wandb:
                wandb.log({
                    'episode': i_episode,
                    'avg_reward': avg_reward
                })
            
            print_running_reward = 0
            print_running_episodes = 0
            
            # Save model
            # --- [핵심] 테스트베드가 사용할 이름(policy_name)으로 저장 ---
            save_path = os.path.join(args.save_dir, args.policy_name)
            ppo_agent.save(save_path)
            print(f"Model saved at {save_path}")

    env.close()
    if args.wandb:
        wandb.finish()
        
if __name__ == '__main__':
    args = get_config()
    main(args)