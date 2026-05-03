"""
Training Script for μ-MuZero
============================
Example usage:
    python train.py --config config.py --steps 2000000 --save_dir checkpoints/
"""

import os
import sys
import argparse
import time
import numpy as np
from collections import deque

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MuMuZeroConfig
from mu_muzero import MuMuZeroAgent


def train(args):
    """Main training loop."""
    config = MuMuZeroConfig()
    
    # Override config from args if provided
    if args.latent_dim:
        config.latent_dim = args.latent_dim
    if args.num_simulations:
        config.num_simulations = args.num_simulations
    
    agent = MuMuZeroAgent(config)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Metrics tracking
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_successes = deque(maxlen=100)
    
    print("=" * 60)
    print("μ-MuZero Training Started")
    print("=" * 60)
    print(f"Device: {agent.device}")
    print(f"Config: latent_dim={config.latent_dim}, "
          f"simulations={config.num_simulations}, "
          f"batch_size={config.batch_size}")
    print(f"Curriculum stages: {len(config.curriculum_stages)}")
    print("=" * 60)
    
    step_count = 0
    episode_count = 0
    start_time = time.time()
    
    while step_count < args.total_steps:
        # Collect episode
        episode_data = agent.run_episode(train=True)
        agent.replay_buffer.append(episode_data)
        
        episode_count += 1
        step_count += len(episode_data['observations'])
        
        # Metrics
        total_reward = sum(episode_data['rewards'])
        success = total_reward > 5.0
        episode_rewards.append(total_reward)
        episode_lengths.append(len(episode_data['observations']))
        episode_successes.append(1.0 if success else 0.0)
        
        # Update curriculum
        agent.update_curriculum()
        
        # Training update
        if len(agent.replay_buffer) >= config.batch_size:
            batch = np.random.choice(
                list(agent.replay_buffer), 
                size=config.batch_size, 
                replace=False
            )
            metrics = agent.train_step(batch)
            
            # Logging
            if step_count % config.log_interval == 0:
                elapsed = time.time() - start_time
                fps = step_count / elapsed
                
                print(f"[Step {step_count:>8}] "
                      f"Episodes: {episode_count:>5} | "
                      f"Reward: {np.mean(episode_rewards):>7.2f} | "
                      f"Success: {np.mean(episode_successes)*100:>5.1f}% | "
                      f"Stage: {agent.current_stage} | "
                      f"Loss: {metrics['total_loss']:.3f} | "
                      f"FPS: {fps:.1f}")
        
        # Checkpoint
        if step_count % config.checkpoint_interval == 0 and step_count > 0:
            ckpt_path = os.path.join(args.save_dir, f"mu_muzero_step_{step_count}.pt")
            agent.save(ckpt_path)
    
    # Final save
    final_path = os.path.join(args.save_dir, "mu_muzero_final.pt")
    agent.save(final_path)
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Total episodes: {episode_count}")
    print(f"Final success rate: {np.mean(episode_successes)*100:.1f}%")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print("=" * 60)


def evaluate(args):
    """Evaluate a trained model."""
    config = MuMuZeroConfig()
    agent = MuMuZeroAgent(config)
    
    # Load checkpoint
    agent.load(args.checkpoint)
    
    print("=" * 60)
    print("μ-MuZero Evaluation")
    print("=" * 60)
    
    num_episodes = args.eval_episodes
    successes = []
    rewards = []
    
    for ep in range(num_episodes):
        episode_data = agent.run_episode(train=False)
        total_reward = sum(episode_data['rewards'])
        success = total_reward > 5.0
        
        successes.append(success)
        rewards.append(total_reward)
        
        print(f"Episode {ep+1}/{num_episodes}: "
              f"Reward={total_reward:.2f}, Success={success}")
    
    print("=" * 60)
    print(f"Success Rate: {np.mean(successes)*100:.1f}%")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="μ-MuZero Training & Evaluation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--total_steps", type=int, default=2000000)
    parser.add_argument("--save_dir", type=str, default="~/klaus_ai/mu-muzero/checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for eval mode")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--num_simulations", type=int, default=None)
    
    args = parser.parse_args()
    
    # Expand user path
    args.save_dir = os.path.expanduser(args.save_dir)
    if args.checkpoint:
        args.checkpoint = os.path.expanduser(args.checkpoint)
    
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if not args.checkpoint:
            print("Error: --checkpoint required for eval mode")
            sys.exit(1)
        evaluate(args)


if __name__ == "__main__":
    main()
