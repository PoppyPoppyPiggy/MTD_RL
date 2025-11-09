# File: MTD_RL/ver_01/train_mtd_only.py
import torch
import numpy as np
from environment import NetworkEnvironment
from ppo import PPO
from metrics import Metrics
import config
from cti_bridge import CTIBridge
from heuristic_seeker import HeuristicSeeker  # 휴리스틱 시커 임포트
import os
import time

def train_mtd_only():
    """
    휴리스틱 공격자(HeuristicSeeker)를 상대로
    방어자(MTDAgent)만 PPO로 학습시키는 스크립트.
    """
    print("Initializing MTD-Only Training...")
    
    # --- 디렉터리 생성 ---
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    
    # --- 환경 및 메트릭 초기화 ---
    # CTIBridge는 environment.py 내부에서 초기화될 수 있음 (원본 코드 확인 필요)
    # 여기서는 CTIBridge를 HeuristicSeeker에 전달하기 위해 별도 초기화
    cti_bridge = CTIBridge() 
    env = NetworkEnvironment(cti_bridge=cti_bridge) # CTI 브리지를 환경에도 전달
    metrics = Metrics(config.LOG_DIR, "mtd_only_training_log")
    
    # --- 에이전트 초기화 ---
    # 1. MTD 에이전트 (PPO로 학습)
    mtd_agent = PPO(
        state_dim=config.STATE_DIM,
        action_dim=config.ACTION_DIM,
        lr_actor=config.LR_ACTOR,
        lr_critic=config.LR_CRITIC,
        gamma=config.GAMMA,
        K_epochs=config.K_EPOCHS,
        eps_clip=config.EPS_CLIP,
        device=config.DEVICE,
        action_std_init=config.ACTION_STD_INIT
    )
    
    # 2. Seeker 에이전트 (휴리스틱)
    seeker = HeuristicSeeker(cti_bridge, config.ACTION_DIM)
    
    print("Agents initialized. Starting training loop...")
    
    start_time = time.time()
    
    # --- 메인 학습 루프 ---
    for episode in range(1, config.MAX_EPISODES + 1):
        state = env.reset()
        episode_mtd_reward = 0
        
        for t in range(config.MAX_TIMESTEPS):
            # --- MTD 에이전트 액션 선택 (학습 대상) ---
            mtd_action = mtd_agent.select_action(state)
            
            # --- Seeker 에이전트 액션 선택 (휴리스틱) ---
            seeker_action = seeker.select_action(state)
            
            # --- 환경 스텝 실행 ---
            # seeker_reward는 무시
            state, (mtd_reward, _), done, info = env.step(mtd_action, seeker_action)
            
            # --- MTD 에이전트 버퍼에 저장 ---
            mtd_agent.buffer.rewards.append(mtd_reward)
            mtd_agent.buffer.is_terminals.append(done)
            
            episode_mtd_reward += mtd_reward
            
            if done:
                break
                
        # --- 에피소드 종료 후 MTD 에이전트 업데이트 ---
        # PPO는 타임스텝마다 업데이트하는 것이 아니라 버퍼가 찰 때 업데이트
        # config.UPDATE_TIMESTEP은 에피소드 수 기준이 아니라, 총 타임스텝 기준일 수 있음.
        # 원본 PPO 구현에 따라 이 로직은 달라질 수 있으나,
        # 여기서는 에피소드 N회마다 업데이트하는 것으로 가정.
        if episode % config.UPDATE_TIMESTEP == 0: 
            print(f"Episode {episode}: Updating MTD agent...")
            mtd_agent.update()
            mtd_agent.clear_buffer() # 업데이트 후 버퍼 비우기

        # --- 로깅 ---
        metrics.log_episode(
            episode=episode,
            mtd_reward=episode_mtd_reward,
            seeker_reward=0,  # 시커는 학습하지 않으므로 0
            avg_mtd_reward=metrics.mtd_rewards.mean(),
            avg_seeker_reward=0
        )
        
        # --- 모델 저장 ---
        if episode % config.SAVE_MODEL_FREQ == 0:
            print(f"\nEpisode {episode}: Saving MTD model...")
            model_save_path = os.path.join(config.MODEL_DIR, "mtd_agent.pth")
            mtd_agent.save_models(model_save_path)
            print(f"Model saved to {model_save_path}")

        # --- 진행 상황 출력 ---
        if episode % 10 == 0:
            metrics.print_status(episode, start_time)

    print("Training finished.")
    env.close()
    metrics.close()

if __name__ == '__main__':
    train_mtd_only()