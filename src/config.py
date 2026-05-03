"""
μ-MuZero Configuration
======================
Hyperparameters and environment settings for micro-scale optical manipulation.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MuMuZeroConfig:
    """Configuration for μ-MuZero algorithm."""
    
    # Environment
    workspace_size: Tuple[float, float, float] = (100.0, 100.0, 50.0)  # μm
    num_traps: int = 3
    max_laser_power: float = 100.0  # mW (hard safety constraint)
    phototoxicity_budget: float = 5000.0  # mW·ms per episode (soft constraint)
    
    # Micro-physics
    viscosity: float = 1.0e-3  # Pa·s (water at 20°C)
    temperature: float = 293.15  # K
    brownian_noise_scale: float = 0.1  # μm / sqrt(ms)
    
    # Neural Networks
    latent_dim: int = 256
    num_res_blocks: int = 8
    hidden_dim: int = 256
    
    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Safety-constrained MCTS
    safety_pruning: bool = True
    uncertainty_penalty_weight: float = 0.5
    
    # Training
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    
    # Self-play
    num_unroll_steps: int = 5
    td_steps: int = 10
    
    # Curriculum
    curriculum_stages: Tuple[dict, ...] = (
        {"name": "single_robot", "success_threshold": 0.7, "max_steps": 50000},
        {"name": "obstacle_avoidance", "success_threshold": 0.7, "max_steps": 200000},
        {"name": "two_robot_coop", "success_threshold": 0.7, "max_steps": 800000},
        {"name": "three_robot_assembly", "success_threshold": 0.7, "max_steps": 2000000},
    )
    
    # Factored action space
    trap_primitives: Tuple[str, ...] = ("hold", "N", "S", "E", "W", "up", "down")
    
    # Logging
    checkpoint_interval: int = 10000
    log_interval: int = 100
