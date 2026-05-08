"""
μ-MuZero: Main Algorithm Interface
==================================
Entry point for training and inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

from config import MuMuZeroConfig
from networks import MuMuZeroModel
from mcts import SafetyConstrainedMCTS
from environment import OpticalMicromanipulationEnv


class MuMuZeroAgent:
    """
    Complete μ-MuZero agent combining model, MCTS, and environment interaction.
    """
    def __init__(self, config: MuMuZeroConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MuMuZeroModel(config).to(self.device)
        self.mcts = SafetyConstrainedMCTS(config, self.model)
        self.env = OpticalMicromanipulationEnv(config)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=100000)
        
        # Curriculum state
        self.current_stage = 0
        self.stage_success_history = deque(maxlen=100)
    
    def select_action(self, obs: np.ndarray, temperature: float = 1.0) -> int:
        """
        Select action using MCTS.
        
        Args:
            obs: microscope image (224, 224)
            temperature: exploration temperature (1.0 for training, 0.0 for eval)
        Returns:
            action_index
        """
        # Convert observation to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Initial inference
        with torch.no_grad():
            latent, policy_logits, value_logits = self.model.initial_inference(obs_tensor)
        
        # Run MCTS
        action_probs = self.mcts.run(latent)
        
        # Sample or greedy
        if temperature > 0:
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(action_probs)
        
        return action
    
    def run_episode(self, train: bool = True) -> Dict:
        """
        Run one episode of self-play.
        
        Returns:
            episode_data containing observations, actions, rewards, policies, values
        """
        obs = self.env.reset()
        done = False
        
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'policies': [],
            'values': [],
        }
        
        while not done:
            # Select action
            temp = 1.0 if train else 0.0
            action = self.select_action(obs['image'], temperature=temp)
            
            # Decode action
            trap_id = action // len(self.config.trap_primitives)
            prim_id = action % len(self.config.trap_primitives)
            
            # Store MCTS policy for training
            with torch.no_grad():
                obs_tensor = torch.tensor(obs['image'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                latent, _, _ = self.model.initial_inference(obs_tensor)
                mcts_policy = self.mcts.run(latent)
            
            episode_data['observations'].append(obs['image'])
            episode_data['actions'].append(action)
            episode_data['policies'].append(mcts_policy)
            
            # Environment step
            obs, reward, done, info = self.env.step((trap_id, prim_id))
            episode_data['rewards'].append(reward)
        
        # Compute n-step returns for value targets
        episode_data['values'] = self._compute_returns(episode_data['rewards'])
        
        # Track success for curriculum
        success = info.get('success', reward > 5.0)
        self.stage_success_history.append(1.0 if success else 0.0)
        
        return episode_data
    
    def _compute_returns(self, rewards: List[float]) -> List[float]:
        """Compute n-step discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.997 * R
            returns.insert(0, R)
        return returns
    
    def update_curriculum(self):
        """Advance curriculum stage if performance threshold is met."""
        if len(self.stage_success_history) < 50:
            return
        
        avg_success = np.mean(self.stage_success_history)
        stage_config = self.config.curriculum_stages[self.current_stage]
        
        if avg_success > stage_config['success_threshold']:
            if self.current_stage < len(self.config.curriculum_stages) - 1:
                self.current_stage += 1
                self.stage_success_history.clear()
                print(f"Curriculum advanced to stage {self.current_stage}: "
                      f"{self.config.curriculum_stages[self.current_stage]['name']}")
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """
        Perform one gradient update on a batch of episodes.
        
        Args:
            batch: list of episode dictionaries
        Returns:
            loss metrics
        """
        # Unpack batch
        obs_batch = []
        target_policies = []
        target_values = []
        target_rewards = []
        actions_batch = []
        
        for episode in batch:
            # Sample a position from the episode
            pos = np.random.randint(0, len(episode['observations']) - self.config.num_unroll_steps)
            
            obs_batch.append(episode['observations'][pos])
            target_policies.append(episode['policies'][pos])
            target_values.append(episode['values'][pos])
            
            # Unroll targets
            for k in range(self.config.num_unroll_steps):
                if pos + k < len(episode['actions']):
                    actions_batch.append(episode['actions'][pos + k])
                    target_rewards.append(episode['rewards'][pos + k])
                else:
                    actions_batch.append(0)
                    target_rewards.append(0.0)
        
        # Convert to tensors
        obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32).unsqueeze(1).to(self.device)
        target_pi = torch.tensor(np.array(target_policies), dtype=torch.float32).to(self.device)
        target_v = torch.tensor(target_values, dtype=torch.float32).to(self.device)
        target_r = torch.tensor(target_rewards, dtype=torch.float32).to(self.device)
        
        # Forward pass: initial inference
        latent, policy_logits, value_logits = self.model.initial_inference(obs_tensor)
        
        # Losses
        policy_loss = -(target_pi * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        
        # Value loss (scalar from categorical support)
        value_support = torch.arange(-300, 301, dtype=torch.float32, device=self.device)
        value_probs = F.softmax(value_logits, dim=-1)
        value_pred = (value_probs * value_support).sum(dim=-1)
        value_loss = F.mse_loss(value_pred, target_v)
        
        # Unroll losses
        reward_loss = 0.0
        for step in range(min(self.config.num_unroll_steps, len(actions_batch) // len(batch))):
            action_idx = actions_batch[step::self.config.num_unroll_steps]
            action_vec = self._get_action_vectors(action_idx)
            
            next_latent, reward_logits, _, _ = self.model.recurrent_inference(latent, action_vec)
            
            reward_probs = F.softmax(reward_logits, dim=-1)
            reward_pred = (reward_probs * value_support).sum(dim=-1)
            reward_loss += F.mse_loss(reward_pred, target_r[step::self.config.num_unroll_steps])
            
            latent = next_latent
        
        # Total loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss +
            self.config.value_loss_weight * value_loss +
            self.config.reward_loss_weight * reward_loss
        )
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'reward_loss': reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss,
        }
    
    def _get_action_vectors(self, action_indices: List[int]) -> torch.Tensor:
        """Convert action indices to network input vectors."""
        vectors = []
        for idx in action_indices:
            trap_id = idx // len(self.config.trap_primitives)
            prim_id = idx % len(self.config.trap_primitives)
            vec = self.model.encode_action(trap_id, prim_id)
            vectors.append(vec)
        return torch.stack(vectors).to(self.device)
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'curriculum_stage': self.current_stage,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_stage = checkpoint.get('curriculum_stage', 0)
        print(f"Checkpoint loaded from {path}")


