# ver_03/ppo.py
# 'rl/ver_02'의 ppo.py를 'rl/ver_03' 로드맵에 맞게 수정
# - PAMDP (Parameterized Action Space)를 지원하도록 Actor-Critic 모델 변경 (로드맵 3단계)

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import numpy as np

# ver_03 config 임포트
from ver_03.config import DEVICE, PPOConfig

config = PPOConfig()

# 공유 네트워크 (Critic과 Actor가 공유하는 FCN)
class SharedNet(nn.Module):
    def __init__(self, state_dim):
        super(SharedNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# Critic 네트워크 (상태 가치 평가)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.shared_net = SharedNet(state_dim)
        self.value_head = nn.Linear(128, 1) # Value
    
    def forward(self, x):
        shared_features = self.shared_net(x)
        value = self.value_head(shared_features)
        return value

# Actor 네트워크 (PAMDP 지원)
class Actor(nn.Module):
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        super(Actor, self).__init__()
        self.shared_net = SharedNet(state_dim)
        
        # 1. Discrete Action Head (이산적 전략 선택)
        self.discrete_head = nn.Linear(128, discrete_action_dim)
        
        # 2. Continuous Action Head (연속적 파라미터 선택)
        self.continuous_mu_head = nn.Linear(128, continuous_action_dim) # 평균(mu)
        # 분산(std)은 학습 가능한 파라미터로 사용 (모든 상태에 대해 동일한 std)
        # 또는 별도 네트워크로 예측 가능. 여기서는 간단하게 nn.Parameter 사용
        self.continuous_log_std = nn.Parameter(torch.zeros(1, continuous_action_dim))

    def forward(self, x):
        shared_features = self.shared_net(x)
        
        # Discrete head
        discrete_logits = self.discrete_head(shared_features)
        
        # Continuous head
        continuous_mu = torch.sigmoid(self.continuous_mu_head(shared_features)) # 0~1 사이 값으로 제한
        continuous_std = torch.exp(self.continuous_log_std.expand_as(continuous_mu)) # 로그 표준편차 -> 표준편차

        return discrete_logits, continuous_mu, continuous_std

class PPO:
    def __init__(self, state_dim, discrete_action_dim, continuous_action_dim):
        self.state_dim = state_dim
        self.discrete_action_dim = discrete_action_dim
        self.continuous_action_dim = continuous_action_dim
        
        self.gamma = config.gamma
        self.k_epochs = config.k_epochs
        self.eps_clip = config.eps_clip
        self.v_clip = config.v_clip
        self.entropy_coef = config.entropy_coef
        self.value_loss_coef = config.value_loss_coef

        # Actor-Critic 모델 생성 (PAMDP)
        self.actor = Actor(state_dim, discrete_action_dim, continuous_action_dim).to(DEVICE)
        self.critic = Critic(state_dim).to(DEVICE)
        
        # Critic 모델의 이전 버전을 저장 (Value Clipping용)
        self.critic_old = Critic(state_dim).to(DEVICE)
        self.critic_old.load_state_dict(self.critic.state_dict())

        # 옵티마이저
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)

        # 버퍼
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            
            # Actor forward
            discrete_logits, continuous_mu, continuous_std = self.actor(state)
            
            # 1. Discrete Action 샘플링
            dist_discrete = Categorical(logits=discrete_logits)
            action_discrete = dist_discrete.sample()
            action_discrete_logprob = dist_discrete.log_prob(action_discrete)
            
            # 2. Continuous Action 샘플링
            dist_continuous = Normal(continuous_mu, continuous_std)
            # 0~1 사이로 클리핑 (환경의 action_space와 일치)
            action_continuous = dist_continuous.sample().clamp(0.0, 1.0) 
            action_continuous_logprob = dist_continuous.log_prob(action_continuous).sum(dim=-1) # 로그 확률 합
            
            # 3. Value 획득
            value = self.critic(state)
            
            # 환경에 전달할 action 딕셔너리
            action_env = {
                "discrete": action_discrete.cpu().item(),
                "continuous": action_continuous.cpu().numpy().flatten()
            }
            
            # 버퍼에 저장할 텐서들
            action_store = {
                "discrete": action_discrete,
                "continuous": action_continuous
            }
            logprob_store = action_discrete_logprob + action_continuous_logprob # 총 로그 확률
            
            return action_env, action_store, logprob_store, value

    def update(self):
        # 1. GAE (Generalized Advantage Estimation) 계산
        rewards = []
        discounted_reward = 0
        log_probs = []
        values = []
        states = []
        actions_discrete = []
        actions_continuous = []

        for r, lp, v, s, a_d, a_c, done in reversed(self.buffer):
            rewards.insert(0, r)
            log_probs.insert(0, lp)
            values.insert(0, v)
            states.insert(0, s)
            actions_discrete.insert(0, a_d)
            actions_continuous.insert(0, a_c)
            
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            # values[i+1]은 마지막 step에서는 0이어야 함
            next_value = values[i+1] if i < len(rewards) - 1 else 0
            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae # 0.95는 GAE lambda
            advantages.insert(0, gae)

        # 텐서 변환
        states = torch.FloatTensor(np.array(states)).to(DEVICE)
        old_log_probs = torch.stack(log_probs, dim=0).to(DEVICE)
        old_values = torch.stack(values, dim=0).to(DEVICE).detach()
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = (advantages + old_values).detach() # GAE + V(s_t) = Q(s_t, a_t) ~ R_t
        
        old_actions_discrete = torch.stack(actions_discrete, dim=0).to(DEVICE).detach()
        old_actions_continuous = torch.stack(actions_continuous, dim=0).to(DEVICE).detach()

        # 2. K-Epochs 만큼 정책 업데이트
        for _ in range(self.k_epochs):
            # --- Actor (Policy) Loss ---
            
            # 현재 정책으로 로그 확률, 엔트로피 계산
            discrete_logits, continuous_mu, continuous_std = self.actor(states)
            
            dist_discrete = Categorical(logits=discrete_logits)
            dist_continuous = Normal(continuous_mu, continuous_std)
            
            new_log_probs_discrete = dist_discrete.log_prob(old_actions_discrete)
            new_log_probs_continuous = dist_continuous.log_prob(old_actions_continuous).sum(dim=-1)
            new_log_probs = new_log_probs_discrete + new_log_probs_continuous
            
            # 엔트로피 (탐험 보너스)
            entropy_discrete = dist_discrete.entropy()
            entropy_continuous = dist_continuous.entropy().sum(dim=-1)
            entropy = (entropy_discrete + entropy_continuous).mean()
            
            # 비율 (Ratio)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())

            # PPO-Clip 목적 함수
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Critic (Value) Loss ---
            
            # Value Clipping (PPO2)
            new_values = self.critic(states).squeeze()
            values_clipped = old_values.squeeze() + torch.clamp(
                new_values - old_values.squeeze(),
                -self.v_clip,
                self.v_clip
            )
            # MSE 손실
            loss_v1 = F.mse_loss(new_values, returns)
            loss_v2 = F.mse_loss(values_clipped, returns)
            critic_loss = torch.max(loss_v1, loss_v2).mean()

            # --- Total Loss 및 업데이트 ---
            loss = actor_loss + (self.value_loss_coef * critic_loss) - (self.entropy_coef * entropy)

            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_actor.step()
            self.optimizer_critic.step()

        # 버퍼 비우기
        self.buffer = []
        
        # Critic_old 업데이트 (업데이트 없음, PPO-Clip v2는 critic_old가 필요 없음)
        # self.critic_old.load_state_dict(self.critic.state_dict())

    def save(self, checkpoint_path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])