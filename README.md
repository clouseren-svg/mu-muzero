# μ-MuZero: Model-Based RL for Autonomous Micro-Robotic Surgery

**μ-MuZero** (pronounced "micro-MuZero") is a safety-constrained, stochastic variant of the MuZero algorithm, specifically designed for autonomous optical micromanipulation in minimally invasive surgery.

---

## Why μ-MuZero?

Optical microrobots can perform cell-level surgical tasks with sub-micron precision, but current systems are **manually operated**. Existing RL algorithms fail at micro-scale due to three unique challenges:

| Challenge | Macro-Scale RL | Micro-Scale (μ-MuZero) |
|-----------|---------------|------------------------|
| **Dynamics** | Deterministic | **Stochastic** (Brownian motion dominates) |
| **Safety** | Soft preference | **Hard constraints** (laser power limits, phototoxicity) |
| **Action Space** | Discrete/continuous | **Combinatorial** (K traps × 7 primitives = 7^K actions) |

μ-MuZero addresses all three through:
1. **Stochastic latent dynamics** — explicit Gaussian uncertainty modelling
2. **Safety-constrained MCTS** — hard pruning of dangerous actions during planning
3. **Factored action representation** — decouples trap selection from movement primitive

---

## Architecture

```
Observation (Microscope Image)
       ↓
[Representation Network h_θ] → Latent State (256-dim)
       ↓
[Safety-Constrained MCTS]
  ├─ Stochastic dynamics sampling (K trajectories)
  ├─ Hard safety pruning (P > P_max removed)
  └─ Uncertainty-aware UCB selection
       ↓
Action (trap_id, primitive_id)
       ↓
[Digital Twin Environment]
  ├─ Learned optical force model (NN)
  ├─ Hydrodynamics (RPY tensor)
  └─ Brownian motion (Euler-Maruyama)
```

---

## Project Structure

```
mu-muzero/
├── src/
│   ├── config.py          # Hyperparameters & environment settings
│   ├── networks.py        # Representation, Dynamics, Prediction networks
│   ├── mcts.py            # Safety-constrained Monte Carlo Tree Search
│   ├── environment.py     # Physics-informed digital twin
│   ├── mu_muzero.py       # Main agent interface
│   ├── train.py           # Training & evaluation script
│   └── ppt_generator.py   # Slide deck generator
├── slides/
│   └── mu_muzero_algorithm.pptx  # 13-slide presentation
├── docs/
│   └── (paper drafts, design notes)
├── assets/
│   └── (figures, diagrams)
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yunxiao-ren/mu-muzero.git
cd mu-muzero

# Install dependencies
pip install torch numpy python-pptx

# Optional: for GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start

### Training

```bash
python src/train.py --mode train --total_steps 2000000 --save_dir checkpoints/
```

### Evaluation

```bash
python src/train.py --mode eval \
    --checkpoint checkpoints/mu_muzero_final.pt \
    --eval_episodes 100
```

### Generate Presentation

```bash
python src/ppt_generator.py
# Output: slides/mu_muzero_algorithm.pptx
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 256 | Latent state dimension |
| `num_simulations` | 50 | MCTS simulations per decision |
| `c_puct` | 1.25 | Exploration constant |
| `batch_size` | 512 | Training batch size |
| `num_unroll_steps` | 5 | Unroll horizon for dynamics loss |
| `max_laser_power` | 100 mW | Hard safety constraint |
| `phototoxicity_budget` | 5000 mW·ms | Soft safety constraint |

---

## Performance Benchmarks

Results from simulation-based validation (digital twin):

| Task | μ-MuZero | Scripted | PPO | Manual | Oracle MPC |
|------|----------|----------|-----|--------|-----------|
| Single-Robot Transport | **91.3%** | 67.2% | 54.8% | 78.0% | 94.1% |
| Obstacle Avoidance | **89.0%** | 58.0% | — | 62.0% | — |
| Cooperative Transport | **78.4%** | 0% | — | 34.0% | 85.7% |
| Micro-Assembly | **68.0%** | 0% | — | 8.0% | — |

*Phototoxicity budget usage: μ-MuZero 45% vs Manual 68% vs Scripted 82%*

---

## Curriculum Learning

Training progresses through automatically managed difficulty stages:

1. **Stage 1** (0–50K steps): Single-robot transport in empty workspace
2. **Stage 2** (50K–250K): Add static obstacles
3. **Stage 3** (250K–1M): Two-robot cooperative pushing
4. **Stage 4** (1M–2M): Three-robot micro-assembly with tight tolerances

---

## Citation

If you use μ-MuZero in your research, please cite:

```bibtex
@phdthesis{ren2026embodied,
  title={Embodied Intelligence for Autonomous Optical Microrobotic Manipulation in Minimally Invasive Surgery},
  author={Ren, Yunxiao},
  school={Imperial College London},
  year={2026}
}
```

---

## License

MIT License — see LICENSE for details.

---

## Contact

- **Author**: Yunxiao Ren
- **Affiliation**: Hamlyn Centre for Robotic Surgery, Imperial College London
- **Email**: yunxiao.ren@imperial.ac.uk
- **Supervisors**: Dr. Dandan Zhang, Dr. Salzitsa Anastasova-Ivanova, Dr. Benny Lo
