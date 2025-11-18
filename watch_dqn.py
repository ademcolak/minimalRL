import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        out = self.forward(obs)
        return out.argmax().item()

def watch_agent(model_path='checkpoints/dqn_cartpole_best.pth', num_episodes=5):
    """Watch the trained agent play"""

    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first by running: python3 dqn.py")
        return

    print(f"📂 Loading model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)

    q = Qnet()
    q.load_state_dict(checkpoint['model_state_dict'])
    q.eval()

    print(f"✅ Model loaded!")
    print(f"📊 Best Score: {checkpoint.get('best_score', 'N/A')}")
    print(f"📝 Episode: {checkpoint.get('episode', 'N/A')}")
    print(f"\n{'='*60}")
    print(f"🎮 Watching agent play for {num_episodes} episodes...")
    print(f"{'='*60}\n")

    # Create environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')

    total_scores = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        done = False
        episode_score = 0
        steps = 0

        while not done:
            # Select action (no exploration, epsilon=0)
            a = q.sample_action(torch.from_numpy(s).float(), epsilon=0.0)
            s_prime, r, done, truncated, info = env.step(a)
            s = s_prime
            episode_score += r
            steps += 1

            if done or truncated:
                break

        total_scores.append(episode_score)
        print(f"Episode {episode+1}/{num_episodes}: Score = {episode_score:.0f}, Steps = {steps}")

    env.close()

    print(f"\n{'='*60}")
    print(f"📊 Average Score: {sum(total_scores)/len(total_scores):.1f}")
    print(f"📊 Max Score: {max(total_scores):.0f}")
    print(f"📊 Min Score: {min(total_scores):.0f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Watch trained DQN agent play CartPole')
    parser.add_argument('--model', type=str, default='checkpoints/dqn_cartpole_best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to watch')

    args = parser.parse_args()

    watch_agent(args.model, args.episodes)
