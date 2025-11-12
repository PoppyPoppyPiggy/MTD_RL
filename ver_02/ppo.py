# File: ver_02/ppo.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Agent Implementation (v22)
- Testbed-Compatible
- (FIX) train_mtd_only.py의 'list-based buffer'와 호환되도록 수정
- (FIX) torch.tensor(list_of_tensors) 오류를 torch.stack()으로 수정
- (KPI) update()가 loss 값들을 반환하도록 수정
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import os

class Buffer:
    """train_mtd_only.py의 학습 루프와 호환되는 간단한 리스트 기반 버퍼"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]

class ActorCritic(nn.Module):
    """PPO Actor-Critic 네트워크 (테스트베드 인터페이스와 호환)"""
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """주어진 상태(state)에 대해 행동(action)과 로그 확률(logprob)을 반환"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob

    def evaluate(self, state, action):
        """주어진 상태(state)와 행동(action)의 가치(value), 로그 확률(logprob), 엔트로피(entropy)를 반환"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return state_value, log_prob, dist_entropy

class PPO:
    """PPO 에이전트 (train_mtd_only.py 호환)"""
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, entropy_coef, device):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        self.buffer = Buffer()
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """환경에서 1스텝 행동 결정 (학습 데이터 수집용)"""
        with torch.no_grad():
            # state가 이미 텐서가 아닐 수 있으므로 변환
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            action, log_prob = self.policy_old.act(state)
            
        self.buffer.states.append(state) # 텐서 자체를 저장
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_prob)
        
        return action, log_prob.item()

    def update(self):
        """PPO 업데이트 (update_timestep마다 호출됨)"""
        
        # 1. Monte Carlo estimate of returns (보상 계산)
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 2. 버퍼 데이터를 텐서로 변환
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # [FIX] torch.tensor()가 텐서 리스트를 처리하지 못하므로 torch.stack() 사용
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()

        # [KPI] Loss 값을 누적할 변수 추가
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # 3. K_epochs 동안 정책 업데이트
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            state_values, logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Advantages 계산 (Rewards - StateValues)
            advantages = rewards - state_values.squeeze().detach()
            # (Advantage 정규화)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs)
            
            # PPO-Clip Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # [KPI] 개별 Loss 계산
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * self.MseLoss(state_values, rewards.unsqueeze(1)) # rewards 차원 맞추기
            entropy_loss = -self.entropy_coef * dist_entropy.mean()

            # Total Loss
            loss = policy_loss + value_loss + entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # [KPI] Loss 값 누적
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += dist_entropy.mean().item()
            
        # 4. Old policy를 새 policy로 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 5. 버퍼 비우기
        self.buffer.clear()
        
        # [KPI] 평균 Loss 값 반환
        avg_policy_loss = total_policy_loss / self.K_epochs
        avg_value_loss = total_value_loss / self.K_epochs
        avg_entropy = total_entropy / self.K_epochs
        
        return avg_policy_loss, avg_value_loss, avg_entropy

    def save(self, checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Model loaded from {checkpoint_path}")