# ver_03/ppo.py
# 'rl/ver_02'의 ppo.py를 'rl/ver_03' 로드맵에 맞게 수정
# - PAMDP (하이브리드 액션) 지원
# - GAE 구현
# - 2025-11-13: select_action에서 .item() 반환 오류 수정 (TypeError 수정)

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np
from collections import deque

# ver_03 config 임포트
from config import PPOConfig, DEVICE

config = PPOConfig()

# --- Actor-Critic 네트워크 정의 ---

class Actor(nn.Module):
    """
    PAMDP (Parameterized Action MDP)를 위한 Actor 네트워크
    - 상태(State)를 입력받아 이산(Discrete) 행동과 연속(Continuous) 행동의 분포를 출력
    - ver_03 로드맵에 따라 Actor를 분리 (Discrete / Continuous)
    """
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(Actor, self).__init__()
        
        # 공통 특징 추출 레이어 (옵션)
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # 1. 이산 행동 정책망 (Discrete Action Policy Head)
        self.discrete_head = nn.Sequential(
            nn.Linear(config.hidden_dim, discrete_action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 2. 연속 행동 정책망 (Continuous Action Policy Head)
        # Mu(평균)와 Std(표준편차)를 출력
        self.continuous_mu_head = nn.Linear(config.hidden_dim, continuous_action_dim)
        self.continuous_std_head = nn.Linear(config.hidden_dim, continuous_action_dim)

    def forward(self, state):
        shared_features = self.shared_layers(state)
        
        # 이산 행동 분포
        discrete_probs = self.discrete_head(shared_features)
        discrete_dist = Categorical(discrete_probs)
        
        # 연속 행동 분포 (평균, 표준편차)
        mu = self.continuous_mu_head(shared_features)
        # 표준편차는 항상 양수여야 하므로 softplus 적용
        std = torch.softplus(self.continuous_std_head(shared_features)) + 1e-5 # 0이 되는 것을 방지
        
        return discrete_dist, mu, std

class Critic(nn.Module):
    """
    Critic 네트워크
    - 상태(State)를 입력받아 상태 가치(Value)를 출력
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(self, state):
        return self.value_net(state)

# --- PPO 버퍼 ---

class Buffer:
    """ PPO 학습을 위한 롤아웃 버퍼 """
    def __init__(self):
        self.buffer = deque()
    
    def append(self, transition):
        # (reward, log_prob, value, state, action_discrete, action_continuous, done)
        self.buffer.append(transition)
        
    def get_batch(self):
        # 버퍼에서 모든 데이터를 추출하고 비움
        batch = list(self.buffer)
        self.buffer.clear()
        
        # 데이터를 각 리스트로 분리
        rewards = [t[0] for t in batch]
        log_probs = [t[1] for t in batch]
        values = [t[2] for t in batch]
        states = [t[3] for t in batch]
        actions_discrete = [t[4] for t in batch]
        actions_continuous = [t[5] for t in batch]
        dones = [t[6] for t in batch]
        
        return rewards, log_probs, values, states, actions_discrete, actions_continuous, dones

    def __len__(self):
        return len(self.buffer)

# --- PPO 에이전트 ---

class PPO:
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        
        self.state_dim = state_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        
        self.gamma = config.gamma
        self.K_epochs = config.K_epochs
        self.eps_clip = config.eps_clip
        self.lambda_gae = config.lambda_gae
        self.T_horizon = config.T_horizon
        
        # Actor-Critic 네트워크 (Old / New)
        # Old는 샘플링(행동 선택)에, New는 업데이트(학습)에 사용
        self.actor_discrete_old = Actor(state_dim, discrete_action_dim, continuous_action_dim).to(DEVICE)
        self.actor_continuous_old = self.actor_discrete_old # PAMDP에서는 Actor가 하나
        self.critic_old = Critic(state_dim).to(DEVICE)
        
        self.actor_discrete_new = Actor(state_dim, discrete_action_dim, continuous_action_dim).to(DEVICE)
        self.actor_continuous_new = self.actor_discrete_new
        self.critic_new = Critic(state_dim).to(DEVICE)
        
        # 초기 정책 동기화
        self.actor_discrete_new.load_state_dict(self.actor_discrete_old.state_dict())
        self.critic_new.load_state_dict(self.critic_old.state_dict())
        
        # 옵티마이저
        self.optimizer_actor = torch.optim.Adam(self.actor_discrete_new.parameters(), lr=config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic_new.parameters(), lr=config.lr_critic)
        
        # 버퍼
        self.buffer = Buffer()
        
        # 손실 함수
        self.mse_loss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # --- Discrete Action ---
            action_discrete_dist, mu, std = self.actor_discrete_old(state_tensor)
            action_discrete = action_discrete_dist.sample()
            log_prob_discrete = action_discrete_dist.log_prob(action_discrete)

            # --- Continuous Action ---
            action_continuous_dist = Normal(mu, std)
            action_continuous = action_continuous_dist.sample()
            # 다차원 continuous action의 log prob을 합산
            log_prob_continuous = action_continuous_dist.log_prob(action_continuous).sum(-1)

            # --- Total Log Prob ---
            # discrete와 continuous log prob 합산
            total_log_prob = log_prob_discrete + log_prob_continuous
            
            # --- Critic Value ---
            value = self.critic_old(state_tensor)

            # 환경에 전달할 action (dict)
            action_env = {
                "discrete": action_discrete.item(),
                "continuous": action_continuous.cpu().numpy()
            }
            
            # 버퍼에 저장할 action (dict)
            action_store = {
                "discrete": action_discrete.cpu().numpy(),
                "continuous": action_continuous.cpu().numpy()
            }

            # [FIX] .item()을 제거하여 Tensor를 그대로 반환 (TypeError 수정)
            return (
                action_env, 
                action_store, 
                total_log_prob.detach(),  # .item() 제거
                value.detach()            # .item() 제거
            )

    def compute_gae(self, rewards, values, dones):
        """ GAE (Generalized Advantage Estimation) 계산 """
        T = len(rewards)
        advantages = torch.zeros(T, 1).to(DEVICE)
        gae = 0.0

        for t in reversed(range(T)):
            # 마지막 스텝(T)의 value는 0이라고 가정 (혹은 bootstrap)
            next_value = values[t+1] if t < T - 1 else 0.0
            
            # (1 - done)을 곱해서 에피소드 종료 시 next_value를 0으로 만듦
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1.0 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages

    def update(self):
        
        # 1. 버퍼에서 데이터 추출
        (
            rewards, 
            log_probs,   # 이제 Tensor 리스트
            values,      # 이제 Tensor 리스트
            states, 
            actions_discrete, 
            actions_continuous, 
            dones
        ) = self.buffer.get_batch()

        # 2. GAE 및 Returns (TD-Lambda) 계산
        
        # [FIX] .tensor() -> .stack() : values가 Tensor 리스트이므로 stack 사용
        values_tensor = torch.stack(values, dim=0).to(DEVICE).squeeze(-1) # (T, 1) -> (T)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(DEVICE) # (T)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(DEVICE) # (T)
        
        # GAE 계산 (T+1개가 아닌 T개이므로 마지막 값 처리 필요)
        advantages = self.compute_gae(rewards_tensor, values_tensor, dones_tensor).detach()
        # returns = GAE + V(s)
        returns = (advantages + values_tensor).detach()
        
        # GAE 정규화 (옵션)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.squeeze(-1) # (T, 1) -> (T)

        # 3. K_epochs 동안 정책 업데이트
        
        # 4. 'Old' 데이터 (학습 시작 전 데이터) 텐서 변환
        # [FIX] .stack() 사용: log_probs가 Tensor 리스트이므로 stack 사용
        old_log_probs = torch.stack(log_probs, dim=0).to(DEVICE).detach().squeeze(-1) # (T)
        
        old_states = torch.FloatTensor(np.array(states)).to(DEVICE) # (T, state_dim)
        old_actions_discrete = torch.LongTensor(np.array(actions_discrete)).to(DEVICE) # (T)
        old_actions_continuous = torch.FloatTensor(np.array(actions_continuous)).to(DEVICE) # (T, continuous_dim)

        
        for _ in range(self.K_epochs):
            
            # 5. 'New' 정책으로 현재 값들 다시 계산 (Evaluate)
            
            # 5.1 Actor (Discrete)
            dist_discrete_new, mu_new, std_new = self.actor_discrete_new(old_states)
            new_log_prob_discrete = dist_discrete_new.log_prob(old_actions_discrete) # (T)
            dist_entropy_discrete = dist_discrete_new.entropy() # (T)
            
            # 5.2 Actor (Continuous)
            dist_continuous_new = Normal(mu_new, std_new)
            # (T, continuous_dim) -> sum -> (T)
            new_log_prob_continuous = dist_continuous_new.log_prob(old_actions_continuous).sum(-1)
            dist_entropy_continuous = dist_continuous_new.entropy().sum(-1) # (T)
            
            # 5.3 Total Log Prob & Entropy
            new_log_probs = new_log_prob_discrete + new_log_prob_continuous # (T)
            dist_entropy = dist_entropy_discrete + dist_entropy_continuous # (T)
            
            # 5.4 Critic
            new_values = self.critic_new(old_states).squeeze(-1) # (T)

            # 6. 비율 (Ratio) 계산
            ratios = torch.exp(new_log_probs - old_log_probs.detach()) # (T)

            # 7. Actor 손실 (Loss) 계산 (PPO-Clip)
            surr1 = ratios * advantages # (T)
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages # (T)
            
            # 최종 Actor 손실 = - (min(surr1, surr2) + Entropy 보너스)
            # Entropy를 추가하여 탐험(Exploration) 장려
            loss_actor = - (torch.min(surr1, surr2) + config.c1 * dist_entropy).mean()
            
            # 8. Critic 손실 (Loss) 계산 (MSE)
            # Value Function Clipping (옵션, PPO 페이퍼에서 제안됨)
            values_clipped = values_tensor + torch.clamp(new_values - values_tensor, -self.eps_clip, self.eps_clip)
            loss_critic1 = self.mse_loss(new_values, returns)
            loss_critic2 = self.mse_loss(values_clipped, returns)
            loss_critic = torch.max(loss_critic1, loss_critic2).mean() * config.c2

            # 9. 총 손실
            loss = loss_actor + loss_critic

            # 10. 옵티마이저 스텝 (업데이트)
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            
            loss.backward()
            
            # Gradient Clipping (옵션)
            nn.utils.clip_grad_norm_(self.actor_discrete_new.parameters(), config.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic_new.parameters(), config.max_grad_norm)

            self.optimizer_actor.step()
            self.optimizer_critic.step()

        # 11. K_epochs 학습 완료 후, Old 정책을 New 정책으로 업데이트
        self.actor_discrete_old.load_state_dict(self.actor_discrete_new.state_dict())
        self.critic_old.load_state_dict(self.critic_new.state_dict())

    def save(self, checkpoint_path):
        # 액터와 크리틱 모델 저장
        # PAMDP이므로 Actor는 하나만 저장
        torch.save({
            'actor_state_dict': self.actor_discrete_old.state_dict(),
            'critic_state_dict': self.critic_old.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, checkpoint_path)
        print(f"PPO (ver_03) model saved to {checkpoint_path}")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        self.actor_discrete_old.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_old.load_state_dict(checkpoint['critic_state_dict'])
        
        self.actor_discrete_new.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_new.load_state_dict(checkpoint['critic_state_dict'])
        
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        
        print(f"PPO (ver_03) model loaded from {checkpoint_path}")