import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

# --- [핵심] train_mtd_only.py의 루프와 호환되는 간단한 (리스트 기반) 버퍼 ---
class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# --- Actor-Critic 네트워크 ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()
        
        self.device = device
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        """주어진 상태(state)에 대해 행동(action)과 로그 확률(log_prob)을 반환"""
        # state가 numpy 배열일 수 있으므로 텐서로 변환
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.item(), action_logprob.detach()

    def evaluate(self, state, action):
        """
        업데이트 시 사용: 
        주어진 상태(state)와 행동(action)의 로그 확률, 상태 가치(value), 분포 엔트로피(entropy) 반환
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

# --- [핵심] train_mtd_only.py의 `__init__` 호출과 호환되는 PPO 클래스 ---
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip, device, entropy_coef):
        
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        # Buffer
        self.buffer = Buffer()
        
        # Policy
        self.policy = ActorCritic(state_dim, action_dim, device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Old Policy (업데이트 비교 대상)
        self.policy_old = ActorCritic(state_dim, action_dim, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        """훈련 중 데이터 수집을 위한 행동 선택"""
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)
        return action, action_logprob

    def update(self):
        # 1. Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # 2. 텐서로 변환
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # (수정) states 텐서 변환 (Array2Tensor와 동일한 로직)
        old_states = torch.squeeze(torch.tensor(self.buffer.states, dtype=torch.float32).to(self.device), 1).detach()
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device).detach()
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32).to(self.device).detach()

        # 3. K_epochs 동안 정책 업데이트
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Advantages
            advantages = rewards - state_values.detach()
            # (정규화 - 옵션)
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # PPO-Clip Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Total Loss
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.MseLoss(state_values, rewards) - \
                   self.entropy_coef * dist_entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 4. Old policy를 새 policy로 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))