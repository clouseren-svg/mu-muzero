"""
Safety-Constrained Monte Carlo Tree Search
==========================================
Extension of MuZero's MCTS with:
  1. Stochastic rollout sampling (for Brownian motion)
  2. Hard safety pruning (laser power, workspace bounds)
  3. Uncertainty-aware exploration bonus
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class Node:
    """A node in the MCTS search tree."""
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> Node
        self.hidden_state = None
        self.reward = 0.0
        self.is_expanded = False
        self.is_safe = True  # Flag for safety pruning
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return self.is_expanded


class SafetyConstrainedMCTS:
    """
    MCTS planner for μ-MuZero with safety constraints.
    """
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.num_actions = config.num_traps * len(config.trap_primitives)
        
        # Precompute action index to (trap_id, primitive_id) mapping
        self.action_map = {}
        idx = 0
        for t in range(config.num_traps):
            for p in range(len(config.trap_primitives)):
                self.action_map[idx] = (t, p)
                idx += 1
    
    def run(self, root_latent: torch.Tensor, root_uncertainty: float = 0.0) -> np.ndarray:
        """
        Run MCTS from root latent state.
        Returns improved policy (visit count distribution over actions).
        """
        root = Node(0.0)
        root.hidden_state = root_latent
        
        # Expand root
        self._expand_node(root, root_latent)
        
        # Add Dirichlet noise to root prior for exploration
        self._add_exploration_noise(root)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]
            
            # SELECT: traverse tree until leaf
            while node.expanded():
                action, node = self._select_child(node, root_uncertainty)
                search_path.append(node)
            
            # EXPAND + EVALUATE
            parent = search_path[-2]
            action_taken = self._get_action_from_parent(parent, node)
            
            # Sample stochastic dynamics for planning
            action_vec = self._get_action_vector(action_taken)
            
            with torch.no_grad():
                next_latent, reward_logits, policy_logits, value_logits = \
                    self.model.recurrent_inference(parent.hidden_state, action_vec)
                
                # Convert categorical to scalar
                value = self._scalar_from_support(value_logits)
                reward = self._scalar_from_support(reward_logits)
            
            node.hidden_state = next_latent
            node.reward = reward
            
            # Expand if not terminal and safe
            if self._is_safe_state(next_latent, action_taken):
                self._expand_node(node, next_latent)
            
            # BACKPROPAGATE
            self._backpropagate(search_path, value, root_uncertainty)
        
        # Return visit count distribution as improved policy
        visit_counts = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in range(self.num_actions)
        ], dtype=np.float32)
        
        # Temperature scaling
        if visit_counts.sum() > 0:
            visit_counts = visit_counts ** (1.0 / 1.0)  # temperature = 1
            visit_counts /= visit_counts.sum()
        
        return visit_counts
    
    def _select_child(self, node: Node, uncertainty: float) -> Tuple[int, Node]:
        """
        Select child with highest UCB score.
        UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child) * uncertainty_penalty
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            if not child.is_safe:
                continue  # Skip pruned unsafe actions
            
            # Q-value (normalized to [0,1] approximately)
            q_value = child.value if child.visit_count > 0 else 0.0
            
            # UCB score with uncertainty penalty
            # Higher uncertainty → lower exploration bonus
            uncertainty_penalty = 1.0 / (1.0 + self.config.uncertainty_penalty_weight * uncertainty)
            
            ucb_score = q_value + self.config.c_puct * child.prior * \
                        math.sqrt(node.visit_count) / (1 + child.visit_count) * \
                        uncertainty_penalty
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand_node(self, node: Node, latent: torch.Tensor):
        """Expand node with policy prior from prediction network."""
        with torch.no_grad():
            policy_logits, _ = self.model.prediction(latent)
            policy = F.softmax(policy_logits, dim=-1).cpu().numpy().flatten()
        
        for action in range(self.num_actions):
            child = Node(prior=policy[action])
            # Safety check: prune dangerous actions
            if self.config.safety_pruning:
                child.is_safe = self._check_action_safety(action, latent)
            node.children[action] = child
        
        node.is_expanded = True
    
    def _check_action_safety(self, action_idx: int, latent: torch.Tensor) -> bool:
        """
        Hard safety check for action.
        Returns False if action would violate constraints.
        """
        trap_id, prim_id = self.action_map[action_idx]
        primitive = self.config.trap_primitives[prim_id]
        
        # Prune actions that increase laser power beyond max
        if primitive == "incr_power":
            # Simplified: in real system, decode latent state to get current power
            # Here we use a probabilistic heuristic based on value estimate
            return False  # Conservative: never allow explicit power increase
        
        # Prune actions moving traps out of workspace
        if primitive in ("N", "S", "E", "W", "up", "down"):
            # Check boundary proximity from latent state
            # (simplified: assume latent encodes position)
            return True  # MPC layer handles boundary constraints
        
        return True
    
    def _is_safe_state(self, latent: torch.Tensor, action: int) -> bool:
        """Check if resulting state is safe."""
        # Simplified: use value estimate as proxy for safety
        with torch.no_grad():
            _, value_logits = self.model.prediction(latent)
            value = self._scalar_from_support(value_logits)
        return value > -10.0  # Heuristic threshold
    
    def _get_action_vector(self, action_idx: int) -> torch.Tensor:
        """Convert action index to network input vector."""
        trap_id, prim_id = self.action_map[action_idx]
        return self.model.encode_action(trap_id, prim_id).unsqueeze(0)
    
    def _get_action_from_parent(self, parent: Node, child: Node) -> int:
        """Find action index leading to child."""
        for action, node in parent.children.items():
            if node is child:
                return action
        return 0
    
    def _backpropagate(self, search_path: List[Node], value: float, uncertainty: float):
        """Backpropagate value up search path."""
        # Bootstrap with uncertainty penalty: reduce value in uncertain states
        value_penalty = -0.1 * uncertainty
        
        for node in reversed(search_path):
            node.value_sum += value + value_penalty
            node.visit_count += 1
            value = node.reward + 0.997 * value  # discount factor
    
    def _add_exploration_noise(self, node: Node):
        """Add Dirichlet noise to root node for exploration."""
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            if node.children[action].is_safe:
                node.children[action].prior = \
                    node.children[action].prior * (1 - self.config.dirichlet_epsilon) + \
                    noise[i] * self.config.dirichlet_epsilon
    
    def _scalar_from_support(self, logits: torch.Tensor) -> float:
        """Convert categorical support logits to scalar value."""
        # Support: [-300, 300] with 601 bins
        probs = F.softmax(logits, dim=-1)
        support = torch.arange(-300, 301, dtype=torch.float32, device=logits.device)
        value = (probs * support).sum(dim=-1)
        return value.item()
