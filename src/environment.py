"""
Physics-Informed Digital Twin for Optical Micromanipulation
===========================================================
Simulates with realistic physics:
  - Optical trapping forces (harmonic trap model)
  - Hydrodynamic interactions (Stokes drag)
  - Brownian motion (Einstein-Smoluchowski, low Reynolds number)
  - Microscopic imaging (PSF + noise)

Physical constants and parameters are realistic for:
  - 2 μm polystyrene beads in water at 20°C
  - 1064 nm optical tweezers at ~50 mW per trap
  - Overdamped regime (Re << 1)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional


class LearnedForceModel(nn.Module):
    """
    Neural network approximation of optical force field.
    Trained offline on FDTD/T-matrix simulations.
    For testing with real physics, we keep the interface but use
    the analytical _optical_force() in the environment instead.
    """
    def __init__(self, input_dim: int = 67, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3D force in pN
        )

        # Fourier feature embeddings for high-frequency force fields
        self.B = nn.Parameter(torch.randn(3, 32) * 2.0, requires_grad=False)

    def forward(self, rel_pos: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rel_pos: relative position (trap - robot), shape (..., 3) in μm
            orientation: robot orientation angles, shape (..., 3) in rad
        Returns:
            force: 3D optical force, shape (..., 3) in pN
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
    Uses realistic low-Reynolds-number physics.
    """
    def __init__(self, config):
        self.config = config
        self.dt = config.dt  # control timestep [s]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Learned force model (kept for compatibility; analytical model used below)
        self.force_model = LearnedForceModel().to(self.device)

        # Precompute physical constants
        self._compute_physics_constants()

        # Environment state
        self.reset()

    def _compute_physics_constants(self):
        """Precompute constants used in every step."""
        cfg = self.config

        # Stokes drag coefficient: γ = 6πηr  [N·s/m]
        r_m = cfg.robot_radius * 1e-6  # radius in meters
        self.gamma_Ns_m = 6.0 * np.pi * cfg.viscosity * r_m

        # Convert to pN·s/μm for convenience
        # 1 N = 10^12 pN, 1 m = 10^6 μm  =>  1 N·s/m = 10^6 pN·s/μm
        self.gamma_pN_s_um = self.gamma_Ns_m * 1e6

        # Diffusion coefficient: D = k_B T / (6πηr) = k_B T / γ  [m²/s]
        self.D_m2_s = cfg.k_B * cfg.temperature / self.gamma_Ns_m
        self.D_um2_s = self.D_m2_s * 1e12  # μm²/s

        # Brownian displacement std per timestep: σ = √(2 D dt)
        self.brownian_std = np.sqrt(2.0 * self.D_um2_s * self.dt)

        # Thermal fluctuation amplitude around trap center: σ_th = √(k_B T / k)
        # k_B T in pN·μm: 1 J = 10^12 pN · 10^6 μm = 10^18 pN·μm
        kBT_pN_um = cfg.k_B * cfg.temperature * 1e18
        if cfg.trap_stiffness > 0:
            self.thermal_sigma = np.sqrt(kBT_pN_um / cfg.trap_stiffness)
        else:
            self.thermal_sigma = 0.0

        # Trap escape distance (typical ~1-2 μm for 2 μm beads @ 50 mW)
        self.trap_escape_dist = 2.0  # μm

        # Photon dose per step [mW·ms]
        self.photon_dose_per_step = cfg.laser_power * self.dt * 1000.0

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Microrobot states
        self.robots = []
        for _ in range(self.config.num_traps):
            robot = {
                'pos': np.random.uniform(10, 90, size=3),  # μm
                'orientation': np.array([0.0, 0.0, 0.0]),  # rad
                'velocity': np.zeros(3),
            }
            self.robots.append(robot)

        # Optical trap positions (initially at workspace center)
        self.traps = [
            np.array([50.0, 50.0, 25.0]) for _ in range(self.config.num_traps)
        ]

        # Target cell (e.g., a biological cell to be manipulated)
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

        # --- 1. Execute primitive: move trap ---
        disp = self.config.max_trap_displacement
        if primitive == "N":
            self.traps[trap_id][1] += disp
        elif primitive == "S":
            self.traps[trap_id][1] -= disp
        elif primitive == "E":
            self.traps[trap_id][0] += disp
        elif primitive == "W":
            self.traps[trap_id][0] -= disp
        elif primitive == "up":
            self.traps[trap_id][2] += disp * 0.5
        elif primitive == "down":
            self.traps[trap_id][2] -= disp * 0.5
        # "hold" does nothing

        # Clamp traps to workspace
        self.traps[trap_id] = np.clip(
            self.traps[trap_id],
            [0, 0, 0],
            list(self.config.workspace_size)
        )

        # --- 2. Update robot dynamics (overdamped, Re << 1) ---
        # Use exact discrete solution of overdamped harmonic oscillator
        # instead of Euler-Maruyama to handle dt >> tau (fast relaxation).
        # x_{n+1} = x_eq + (x_n - x_eq)·exp(-dt/τ) + σ_eq·√(1-exp(-2dt/τ))·N(0,1)
        for i, robot in enumerate(self.robots):
            rel_pos = self.traps[i] - robot['pos']  # μm
            force = self._optical_force(rel_pos)  # pN

            # Free diffusion if no trap force (outside capture range)
            if np.linalg.norm(force) < 1e-6:
                brownian = np.random.normal(0.0, self.brownian_std, size=3)
                robot['pos'] += brownian
                robot['velocity'] = brownian / self.dt
                robot['pos'] = np.clip(
                    robot['pos'], [0, 0, 0], list(self.config.workspace_size)
                )
                continue

            # Effective stiffness per direction (from F = k·Δx)
            k_per_dir = np.abs(force) / (np.abs(rel_pos) + 1e-8)  # pN/μm

            # Relaxation time τ = γ / k  [s]
            tau_per_dir = self.gamma_pN_s_um / k_per_dir
            relax_per_dir = np.exp(-self.dt / tau_per_dir)

            # Thermal equilibrium fluctuation amplitude σ_eq = √(k_B T / k)
            kBT = self.config.k_B * self.config.temperature * 1e18  # pN·μm
            sigma_eq_per_dir = np.sqrt(kBT / k_per_dir)  # μm

            # Discrete update per dimension
            new_pos = np.zeros(3)
            for dim in range(3):
                x_eq = self.traps[i][dim]
                relax = relax_per_dir[dim]
                sigma_eq = sigma_eq_per_dir[dim]

                # Correlated thermal noise for discrete harmonic oscillator
                noise_std = sigma_eq * np.sqrt(1.0 - relax**2)
                noise = np.random.normal(0.0, noise_std)

                new_pos[dim] = x_eq + (robot['pos'][dim] - x_eq) * relax + noise

            # Effective velocity for observation
            robot['velocity'] = (new_pos - robot['pos']) / self.dt
            robot['pos'] = new_pos

            # Clamp to workspace
            robot['pos'] = np.clip(
                robot['pos'], [0, 0, 0], list(self.config.workspace_size)
            )

        # --- 3. Update target if being pushed by robots ---
        self._update_target_dynamics()

        # --- 4. Update phototoxicity budget ---
        self.phototoxicity += self.photon_dose_per_step

        # --- 5. Compute reward ---
        reward = self._compute_reward()

        # --- 6. Check termination ---
        done = self._check_done()
        self.step_count += 1

        info = {
            'phototoxicity': self.phototoxicity,
            'step_count': self.step_count,
            'brownian_std': self.brownian_std,
            'trap_stiffness': self.config.trap_stiffness,
        }

        return self._get_observation(), reward, done, info

    def _optical_force(self, rel_pos: np.ndarray) -> np.ndarray:
        """
        Analytical harmonic trap force model.

        F = k · (r_trap - r_particle)   within capture range
        F = 0                             beyond capture range

        Axial stiffness is ~1/3 of lateral (typical for optical tweezers).

        Args:
            rel_pos: trap - particle position vector [μm]
        Returns:
            force: 3D optical force [pN]
        """
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-6 or dist > self.trap_escape_dist:
            return np.zeros(3)

        k = self.config.trap_stiffness  # pN/μm

        # Harmonic restoring force
        force = k * rel_pos  # pN

        # Axial weakening (z-direction stiffness ~ 1/3 of lateral)
        force[2] *= 0.3

        # Soft truncation at escape distance (prevents sudden jumps)
        soften = 1.0 - (dist / self.trap_escape_dist)
        force *= soften

        return force

    def _update_target_dynamics(self):
        """Update target cell position based on robot contacts."""
        contact_force = np.zeros(3)
        for robot in self.robots:
            dist = np.linalg.norm(robot['pos'] - self.target['pos'])
            if dist < (self.config.robot_radius + self.target['radius']):
                # Contact: push target
                direction = robot['pos'] - self.target['pos']
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                # Contact force scaled by overlap depth (Hertz-like)
                overlap = (self.config.robot_radius + self.target['radius']) - dist
                contact_force += direction * overlap * 20.0  # pN/μm of overlap

        # Apply to target (larger object, higher drag)
        gamma_target = 6 * np.pi * self.config.viscosity * self.target['radius'] * 1e-6
        gamma_target_pN_s_um = gamma_target * 1e6

        velocity = contact_force / gamma_target_pN_s_um * self.dt  # μm per step
        self.target['pos'] += velocity
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
