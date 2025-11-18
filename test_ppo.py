"""
Test script for trained PPO model
Loads the best saved model and runs test episodes
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

# Same network architecture as training
class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def sample_action(self, obs):
        """Sample action from policy"""
        prob = self.pi(obs)
        m = Categorical(prob)
        return m.sample().item()

def test_model(model_path='checkpoints/ppo_cartpole_best.pth',
               num_episodes=10,
               render=False):
    """
    Test a trained PPO model

    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of test episodes
        render: Whether to render the environment (requires display)
    """

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first by running: python3 ppo.py")
        return

    # Load model
    print(f"📂 Loading model: {model_path}")
    model = PPO()
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    print(f"✅ Model loaded!")
    print(f"   Episode: {checkpoint.get('episode', 'N/A')}")
    print(f"   Best Score: {checkpoint.get('best_score', 'N/A'):.1f}")
    print(f"\n{'='*60}")
    print(f"🎮 Running {num_episodes} test episodes...")
    print(f"{'='*60}\n")

    # Create environment
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')

    scores = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        done = False
        score = 0
        steps = 0

        while not done:
            # Sample action from policy (no exploration)
            with torch.no_grad():
                a = model.sample_action(torch.from_numpy(s).float())

            s, r, done, truncated, info = env.step(a)
            score += r
            steps += 1

            # Episode ends if either done OR truncated
            if done or truncated:
                break

        scores.append(score)
        print(f"Episode {episode+1:2d}: Score = {score:6.1f}, Steps = {steps:3d}")

    env.close()

    # Statistics
    print(f"\n{'='*60}")
    print(f"📊 Test Results:")
    print(f"{'='*60}")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Min Score:     {np.min(scores):.1f}")
    print(f"Max Score:     {np.max(scores):.1f}")
    print(f"Success Rate:  {sum(s >= 195 for s in scores)/len(scores)*100:.1f}%")
    print(f"{'='*60}\n")

    if np.mean(scores) >= 195:
        print("✅ Model is performing well! (avg >= 195)")
    else:
        print("⚠️  Model needs more training (avg < 195)")

def compare_models():
    """Compare best model vs latest model"""
    best_path = 'checkpoints/ppo_cartpole_best.pth'
    latest_path = 'checkpoints/ppo_cartpole_latest.pth'

    print("\n🔍 Comparing Models...")
    print(f"{'='*60}\n")

    if os.path.exists(best_path):
        print("📊 Best Model:")
        test_model(best_path, num_episodes=10, render=False)

    if os.path.exists(latest_path):
        print("\n📊 Latest Model:")
        test_model(latest_path, num_episodes=10, render=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained PPO model')
    parser.add_argument('--model', type=str, default='checkpoints/ppo_cartpole_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment (requires display)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare best vs latest model')

    args = parser.parse_args()

    if args.compare:
        compare_models()
    else:
        test_model(args.model, args.episodes, args.render)
