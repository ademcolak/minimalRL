"""
Test script for trained DQN model
Loads the best saved model and runs test episodes
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Same network architecture as training
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon=0.0):
        """Sample action (epsilon=0 for greedy policy)"""
        out = self.forward(obs)
        return out.argmax().item()

def test_model(model_path='checkpoints/dqn_cartpole_best.pth',
               num_episodes=10,
               render=False):
    """
    Test a trained DQN model

    Args:
        model_path: Path to saved model checkpoint
        num_episodes: Number of test episodes
        render: Whether to render the environment (requires display)
    """

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first by running: python3 dqn.py")
        return

    # Load model
    print(f"📂 Loading model: {model_path}")
    q = Qnet()
    checkpoint = torch.load(model_path)
    q.load_state_dict(checkpoint['model_state_dict'])
    q.eval()  # Set to evaluation mode

    print(f"✅ Model loaded!")
    print(f"   Episode: {checkpoint['episode']}")
    print(f"   Best Score: {checkpoint['best_score']:.1f}")
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
            # Greedy action selection (no exploration)
            with torch.no_grad():
                a = q.sample_action(torch.from_numpy(s).float(), epsilon=0.0)

            s, r, done, truncated, info = env.step(a)
            score += r
            steps += 1

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
    best_path = 'checkpoints/dqn_cartpole_best.pth'
    latest_path = 'checkpoints/dqn_cartpole_latest.pth'

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

    parser = argparse.ArgumentParser(description='Test trained DQN model')
    parser.add_argument('--model', type=str, default='checkpoints/dqn_cartpole_best.pth',
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
