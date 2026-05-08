"""
Neural Networks for μ-MuZero
============================
Three networks:
  - Representation h_θ: observation → latent state
  - Dynamics g_θ: (latent_state, action) → (next_latent, reward)
  - Prediction f_θ: latent_state → (policy, value)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class RepresentationNetwork(nn.Module):
    """
    h_θ: Encodes microscope image + belief into latent state.
    Input: (B, C, H, W) image tensor
    Output: (B, latent_dim) latent state vector
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 256, num_blocks: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial convolution for micro-scale images (224x224 grayscale)
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Residual tower
        self.res_tower = nn.Sequential(*[
            ResidualBlock(256) for _ in range(num_blocks)
        ])
        
        # Global average pooling + projection to latent
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(obs)
        x = self.res_tower(x)
        x = self.projection(x)
        return x


class DynamicsNetwork(nn.Module):
    """
    g_θ: Predicts next latent state and reward.
    Stochastic variant: outputs Gaussian parameters (μ, logσ) for next state.
    
    Input: (B, latent_dim + action_dim)
    Output: 
      - next_state_mu: (B, latent_dim)
      - next_state_logsigma: (B, latent_dim)
      - reward: (B, support_size)
    """
    def __init__(self, latent_dim: int = 256, action_dim: int = 7, 
                 reward_support_size: int = 601):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.reward_support_size = reward_support_size
        
        input_dim = latent_dim + action_dim
        hidden_dim = 256
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Stochastic dynamics: predict mean and log-std
        self.next_state_mu = nn.Linear(hidden_dim, latent_dim)
        self.next_state_logsigma = nn.Linear(hidden_dim, latent_dim)
        
        # Reward prediction (categorical over support)
        self.reward_head = nn.Linear(hidden_dim, reward_support_size)
        
        # Learned state normalization (as in MuZero)
        # Using LayerNorm instead of BatchNorm1d to support batch_size=1 during inference
        self.state_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = torch.cat([latent, action], dim=-1)
        x = self.mlp(x)
        
        next_mu = self.next_state_mu(x)
        next_logsigma = self.next_state_logsigma(x)
        next_logsigma = torch.clamp(next_logsigma, min=-10, max=2)
        
        reward_logits = self.reward_head(x)
        
        return next_mu, next_logsigma, reward_logits
    
    def sample_next_state(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Sample a next state from the predicted distribution."""
        mu, logsigma, _ = self.forward(latent, action)
        sigma = torch.exp(logsigma)
        eps = torch.randn_like(mu)
        next_state = mu + sigma * eps
        next_state = self.state_norm(next_state)
        return next_state


class PredictionNetwork(nn.Module):
    """
    f_θ: Predicts policy and value from latent state.
    
    Input: (B, latent_dim)
    Output:
      - policy_logits: (B, num_actions)
      - value_logits: (B, support_size)
    """
    def __init__(self, latent_dim: int = 256, num_actions: int = 10,
                 value_support_size: int = 601):
        super().__init__()
        hidden_dim = 256
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, value_support_size)
    
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(latent)
        policy_logits = self.policy_head(x)
        value_logits = self.value_head(x)
        return policy_logits, value_logits


class MuMuZeroModel(nn.Module):
    """
    Complete μ-MuZero model combining all three networks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Factored action space: num_traps * (1 + 6 primitives)
        # We encode action as (trap_id_one_hot, primitive_one_hot)
        self.action_dim = config.num_traps + len(config.trap_primitives)
        self.num_actions = config.num_traps * len(config.trap_primitives)
        
        self.representation = RepresentationNetwork(
            in_channels=1,
            latent_dim=config.latent_dim,
            num_blocks=config.num_res_blocks
        )
        self.dynamics = DynamicsNetwork(
            latent_dim=config.latent_dim,
            action_dim=self.action_dim,
        )
        self.prediction = PredictionNetwork(
            latent_dim=config.latent_dim,
            num_actions=self.num_actions,
        )
    
    def encode_action(self, trap_id: int, primitive_id: int) -> torch.Tensor:
        """Encode factored action into continuous vector."""
        trap_onehot = F.one_hot(
            torch.tensor(trap_id), num_classes=self.config.num_traps
        ).float()
        prim_onehot = F.one_hot(
            torch.tensor(primitive_id), num_classes=len(self.config.trap_primitives)
        ).float()
        return torch.cat([trap_onehot, prim_onehot], dim=-1)
    
    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h_θ + f_θ: from observation to latent, policy, value."""
        latent = self.representation(obs)
        policy_logits, value_logits = self.prediction(latent)
        return latent, policy_logits, value_logits
    
    def recurrent_inference(self, latent: torch.Tensor, action_vec: torch.Tensor) -> Tuple:
        """g_θ + f_θ: from (latent, action) to next_latent, reward, policy, value."""
        next_mu, next_logsigma, reward_logits = self.dynamics(latent, action_vec)
        
        # For planning, we use the mean prediction (deterministic abstraction)
        next_latent = self.dynamics.state_norm(next_mu)
        
        policy_logits, value_logits = self.prediction(next_latent)
        return next_latent, reward_logits, policy_logits, value_logits
