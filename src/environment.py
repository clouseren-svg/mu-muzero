"""
Physics-Informed Digital Twin for Optical Micromanipulation
===========================================================
Simulates:
  - Optical trapping forces (learned NN approximation)
  - Hydrodynamic interactions (RPY tensor)
  - Brownian motion (Euler-Maruyama)
  - Microscopic imaging (PSF + noise)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List


class LearnedForceModel(nn.Module):
    """
    Neural network approximation of optical force field.
    Trained offline on FDTD/T-matrix simulations.
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3D force
        )
        
        # Fourier feature embeddings for high-frequency force fields
        self.B = nn.Parameter(torch.randn(3, 32) * 2.0, requires_grad=False)
    
    def forward(self, rel_pos: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: relative position (trap - robot), shape (..., 3)
            orientation: robot orientation angles, shape (..., 3)
        Returns:
            force: 3D optical force, shape (..., 3)
        """
        # Fourier features for position
        fourier = torch.cat([
            torch.sin(2 * np.pi * rel_pos @ self.B),
            torch.cos(2 * np.pi * rel_pos @ self.B)
        ], dim=-1)
        
        x = torch.cat([fourier, orientation], dim=-1)
        force = self.net(x)
        return force


class OpticalMicromanipulationEnv:
    """
    Digital twin environment for optical microrobot manipulation.
    """
    def __init__(self, config):
        self.config = config
        self.dt = 0.05  # 50 ms control timestep
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Learned force model
        self.force_model = LearnedForceModel().to(self.device)
        
        # Environment state
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        # Microrobot states: [x, y, z, phi, theta, psi, vx, vy, vz, ...]
        self.robots = []
        for _ in range(self.config.num_traps):  # Using num_traps as num_robots for simplicity
            robot = {
                'pos': np.random.uniform(10, 90, size=3),  # μm
                'orientation': np.array([0.0, 0.0, 0.0]),  # rad
                'velocity': np.zeros(3),
            }
            self.robots.append(robot)
        
        # Optical trap positions
        self.traps = [
            np.array([50.0, 50.0, 25.0]) for _ in range(self.config.num_traps)
        ]
        
        # Target cell
        self.target = {
            'pos': np.random.uniform(30, 70, size=3),
            'radius': 5.0,  # μm
        }
        
        # Goal position
        self.goal = np.array([80.0, 80.0, 25.0])
        
        # Cumulative metrics
        self.phototoxicity = 0.0
        self.step_count = 0
        
        return self._get_observation()
    
    def step(self, action: Tuple[int, int]) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one control step.
        
        Args:
            action: (trap_id, primitive_id)
        Returns:
            observation, reward, done, info
        """
        trap_id, prim_id = action
        primitive = self.config.trap_primitives[prim_id]
        
        # Execute primitive
        if primitive == "N":
            self.traps[trap_id][1] += 2.0
        elif primitive == "S":
            self.traps[trap_id][1] -= 2.0
        elif primitive == "E":
            self.traps[trap_id][0] += 2.0
        elif primitive == "W":
            self.traps[trap_id][0] -= 2.0
        elif primitive == "up":
            self.traps[trap_id][2] += 1.0
        elif primitive == "down":
            self.traps[trap_id][2] -= 1.0
        # "hold" does nothing
        
        # Clamp traps to workspace
        self.traps[trap_id] = np.clip(
            self.traps[trap_id], 
            [0, 0, 0], 
            list(self.config.workspace_size)
        )
        
        # Update robot dynamics (simplified low Reynolds number)
        for i, robot in enumerate(self.robots):
            # Optical force from nearest trap
            rel_pos = self.traps[i] - robot['pos']
            rel_pos_t = torch.tensor(rel_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
            orient_t = torch.tensor(robot['orientation'], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                force = self.force_model(rel_pos_t, orient_t).cpu().numpy().flatten()
            
            # Hydrodynamic drag (Stokes drag)
            gamma = 6 * np.pi * self.config.viscosity * 2.0  # 2 μm radius
            
            # Brownian motion
            brownian = np.random.normal(0, self.config.brownian_noise_scale, size=3)
            
            # Overdamped dynamics: γ v = F + F_brownian
            velocity = (force + brownian) / gamma * 1e6  # Convert to μm/s
            robot['velocity'] = velocity
            robot['pos'] += velocity * self.dt
            
            # Clamp to workspace
            robot['pos'] = np.clip(robot['pos'], [0, 0, 0], list(self.config.workspace_size))
        
        # Update target if being pushed by robots
        self._update_target_dynamics()
        
        # Update phototoxicity budget
        laser_power = 50.0  # mW per trap (simplified)
        self.phototoxicity += laser_power * self.dt * 1000  # mW·ms
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        done = self._check_done()
        
        self.step_count += 1
        
        info = {
            'phototoxicity': self.phototoxicity,
            'step_count': self.step_count,
        }
        
        return self._get_observation(), reward, done, info
    
    def _update_target_dynamics(self):
        """Update target cell position based on robot contacts."""
        contact_force = np.zeros(3)
        for robot in self.robots:
            dist = np.linalg.norm(robot['pos'] - self.target['pos'])
            if dist < (2.0 + self.target['radius']):
                # Contact: push target
                direction = self.target['pos'] - robot['pos']
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                contact_force += direction * 10.0  # pN
        
        # Apply to target (larger object, higher drag)
        gamma_target = 6 * np.pi * self.config.viscosity * self.target['radius']
        velocity = contact_force / gamma_target * 1e6
        self.target['pos'] += velocity * self.dt
        self.target['pos'] = np.clip(
            self.target['pos'], 
            [0, 0, 0], 
            list(self.config.workspace_size)
        )
    
    def _compute_reward(self) -> float:
        """Compute step reward."""
        # Task reward: negative distance to goal
        dist_to_goal = np.linalg.norm(self.target['pos'] - self.goal)
        task_reward = -dist_to_goal * 0.1
        
        # Efficiency penalty
        efficiency_penalty = -0.01 * self.step_count
        
        # Safety penalty
        safety_penalty = 0.0
        if self.phototoxicity > self.config.phototoxicity_budget:
            safety_penalty = -10.0
        
        # Success bonus
        if dist_to_goal < 2.0:
            task_reward += 10.0
        
        return task_reward + efficiency_penalty + safety_penalty
    
    def _check_done(self) -> bool:
        """Check if episode is terminated."""
        # Success: target within 2 μm of goal
        if np.linalg.norm(self.target['pos'] - self.goal) < 2.0:
            return True
        
        # Failure: phototoxicity exceeded or timeout
        if self.phototoxicity > self.config.phototoxicity_budget * 1.5:
            return True
        if self.step_count > 500:
            return True
        
        return False
    
    def _get_observation(self) -> Dict:
        """Generate microscope image observation."""
        # Simplified: render 224x224 grayscale image
        img = np.zeros((224, 224), dtype=np.float32)
        
        # Scale factor: workspace (100x100 μm) -> image (224x224 px)
        scale = 224.0 / 100.0
        
        def world_to_img(pos):
            return int(pos[0] * scale), int(pos[1] * scale)
        
        # Draw robots
        for robot in self.robots:
            x, y = world_to_img(robot['pos'])
            if 0 <= x < 224 and 0 <= y < 224:
                # Draw Gaussian blob
                yy, xx = np.mgrid[0:224, 0:224]
                sigma = 3.0
                blob = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                img += blob * 0.8
        
        # Draw target
        x, y = world_to_img(self.target['pos'])
        if 0 <= x < 224 and 0 <= y < 224:
            yy, xx = np.mgrid[0:224, 0:224]
            sigma = 8.0
            blob = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
            img += blob * 0.6
        
        # Add sensor noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)
        
        return {
            'image': img,
            'goal': self.goal.copy(),
            'phototoxicity': self.phototoxicity,
        }
    
    def render(self):
        """Render environment state (for debugging)."""
        obs = self._get_observation()
        return obs['image']
