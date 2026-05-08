"""
Test script for μ-MuZero — runs with synthetic/virtual data.
"""

import sys
import torch
import numpy as np

from config import MuMuZeroConfig
from mu_muzero import MuMuZeroAgent


def test_environment():
    """Test environment initialization and one episode."""
    print("\n" + "=" * 60)
    print("Test 1: Environment")
    print("=" * 60)

    from environment import OpticalMicromanipulationEnv
    config = MuMuZeroConfig()
    env = OpticalMicromanipulationEnv(config)

    obs = env.reset(seed=42)
    print(f"  Observation keys: {obs.keys()}")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Goal: {obs['goal']}")
    print(f"  Phototoxicity: {obs['phototoxicity']}")

    total_reward = 0
    for step in range(10):
        action = (0, 1)  # trap 0, move N
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"  Ran {step + 1} steps, total reward: {total_reward:.3f}")
    print("  [PASS]")


def test_model_inference():
    """Test model forward passes."""
    print("\n" + "=" * 60)
    print("Test 2: Model Inference")
    print("=" * 60)

    from networks import MuMuZeroModel
    config = MuMuZeroConfig()
    model = MuMuZeroModel(config)
    device = torch.device("cpu")
    model = model.to(device)

    # Initial inference
    obs = torch.randn(2, 1, 224, 224).to(device)
    latent, policy_logits, value_logits = model.initial_inference(obs)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value logits shape: {value_logits.shape}")

    # Recurrent inference
    action_vec = model.encode_action(0, 1).unsqueeze(0).repeat(2, 1).to(device)
    next_latent, reward_logits, policy_logits, value_logits = model.recurrent_inference(latent, action_vec)
    print(f"  Next latent shape: {next_latent.shape}")
    print(f"  Reward logits shape: {reward_logits.shape}")
    print("  [PASS]")


def test_mcts():
    """Test MCTS planner."""
    print("\n" + "=" * 60)
    print("Test 3: MCTS")
    print("=" * 60)

    from mcts import SafetyConstrainedMCTS
    from networks import MuMuZeroModel

    config = MuMuZeroConfig()
    config.num_simulations = 10  # fewer for speed
    model = MuMuZeroModel(config)
    mcts = SafetyConstrainedMCTS(config, model)

    latent = torch.randn(1, config.latent_dim)
    policy = mcts.run(latent)
    print(f"  Policy shape: {policy.shape}")
    print(f"  Policy sum: {policy.sum():.4f}")
    print(f"  Best action: {policy.argmax()}")
    print("  [PASS]")


def test_agent_episode():
    """Test full agent running one episode."""
    print("\n" + "=" * 60)
    print("Test 4: Agent Episode (1 episode, few simulations)")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.num_simulations = 5  # very few for speed
    config.num_traps = 2
    config.trap_primitives = ("hold", "N", "S", "E", "W")
    agent = MuMuZeroAgent(config)

    episode_data = agent.run_episode(train=True)
    print(f"  Episode length: {len(episode_data['observations'])}")
    print(f"  Total reward: {sum(episode_data['rewards']):.3f}")
    print(f"  Num observations: {len(episode_data['observations'])}")
    print(f"  Num actions: {len(episode_data['actions'])}")
    print(f"  Num policies: {len(episode_data['policies'])}")
    print(f"  Num values: {len(episode_data['values'])}")
    print("  [PASS]")


def test_training_step():
    """Test training step with synthetic episode batch."""
    print("\n" + "=" * 60)
    print("Test 5: Training Step")
    print("=" * 60)

    config = MuMuZeroConfig()
    config.num_simulations = 3
    config.num_traps = 2
    config.trap_primitives = ("hold", "N", "S", "E", "W")
    config.batch_size = 4
    config.num_unroll_steps = 3
    agent = MuMuZeroAgent(config)

    # Generate synthetic episodes
    episodes = []
    for _ in range(8):
        ep = agent.run_episode(train=True)
        episodes.append(ep)

    batch = np.random.choice(episodes, size=config.batch_size, replace=False)
    metrics = agent.train_step(batch.tolist())
    print(f"  Total loss: {metrics['total_loss']:.4f}")
    print(f"  Policy loss: {metrics['policy_loss']:.4f}")
    print(f"  Value loss: {metrics['value_loss']:.4f}")
    print(f"  Reward loss: {metrics['reward_loss']:.4f}")
    print("  [PASS]")


def test_save_load():
    """Test checkpoint save/load."""
    print("\n" + "=" * 60)
    print("Test 6: Save/Load")
    print("=" * 60)

    import tempfile
    import os

    config = MuMuZeroConfig()
    config.num_simulations = 3
    config.num_traps = 1
    config.trap_primitives = ("hold", "N")
    agent = MuMuZeroAgent(config)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        agent.save(path)
        agent.load(path)
        print("  Save/load OK")
        print("  [PASS]")
    finally:
        os.unlink(path)


def main():
    print("\n" + "=" * 60)
    print("μ-MuZero Integration Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    try:
        test_environment()
        test_model_inference()
        test_mcts()
        test_agent_episode()
        test_training_step()
        test_save_load()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n[FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
