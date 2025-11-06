#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Agent Implementation (ActorCritic, Buffer, PPO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Buffer:
    """PPO 롤아웃 버퍼"""
    def __init__(self, n_envs: int, rollout_steps: int, state_dim: int, device: torch.device):
        self.states = torch.zeros((rollout_steps, n_envs, state_dim), device=device)
        self.actions = torch.zeros((rollout_steps, n_envs), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((rollout_steps, n_envs), device=device)
        self.rewards = torch.zeros((rollout_steps, n_envs), device=device)
        self.dones = torch.zeros((rollout_steps, n_envs), device=device)
        self.values = torch.zeros((rollout_steps, n_envs), device=device)
        self.rollout_steps, self.n_envs, self.device = rollout_steps, n_envs, device
        self.ptr = 0

    def add(self, s,a,r,d,logp,v):
        """버퍼에 1 스텝 데이터 추가"""
        self.states[self.ptr] = s; self.actions[self.ptr] = a; self.rewards[self.ptr] = r
        self.dones[self.ptr] = d; self.log_probs[self.ptr] = logp; self.values[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.rollout_steps

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """GAE (Generalized Advantage Estimation) 계산"""
        advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal, next_values = 1.0 - self.dones[t], last_value
            else:
                next_non_terminal, next_values = 1.0 - self.dones[t], self.values[t+1]
            delta = self.rewards[t] + gamma*next_values*next_non_terminal - self.values[t]
            advantages[t] = last_gae_lam = delta + gamma*gae_lambda*next_non_terminal*last_gae_lam
        returns = advantages + self.values
        return returns, advantages

    def get(self):
        """버퍼의 모든 데이터를 flat 텐서로 반환"""
        states = self.states.view(-1, self.states.size(-1))
        actions = self.actions.flatten()
        log_probs = self.log_probs.flatten()
        return states, actions, log_probs

class ActorCritic(nn.Module):
    """Actor-Critic 신경망"""
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh())
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, s):
        x = self.shared(s)
        return Categorical(logits=self.actor(x)), self.critic(x).squeeze(-1)

    def act(self, s):
        dist, v = self.forward(s)
        a = dist.sample()
        return a, dist.log_prob(a), v

class PPO:
    """PPO 에이전트 (업데이트 로직 포함)"""
    def __init__(self, state_dim, action_dim, device, lr, eps_clip, entropy_coef):
        self.device, self.eps_clip, self.entropy_coef = device, eps_clip, entropy_coef
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, buf: Buffer, returns: torch.Tensor, adv: torch.Tensor, k_epochs: int, max_grad_norm: float):
        """PPO 클리핑을 사용한 정책 및 가치 함수 업데이트"""
        s, a, old_lp = buf.get()
        returns, adv = returns.flatten(), adv.flatten()
        
        # Advantage 정규화 (Zero mean, unit variance) - 중요!
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        pl, vl, ent = 0.,0.,0.
        for _ in range(k_epochs):
            dist, values = self.policy(s)
            lp = dist.log_prob(a); entropy = dist.entropy().mean()
            
            # Policy Loss (Clip)
            ratios = torch.exp(lp - old_lp)
            s1 = ratios * adv
            s2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            policy_loss = -torch.min(s1, s2).mean()
            
            # Value Loss (MSE)
            value_loss = F.mse_loss(values, returns)

            # Total Loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.opt.step()
            
            pl += policy_loss.item(); vl += value_loss.item(); ent += entropy.item()
            
        return [pl/k_epochs, vl/k_epochs, ent/k_epochs]