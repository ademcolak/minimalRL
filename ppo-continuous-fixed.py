import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import collections
import numpy as np
import os
from datetime import datetime

# Hyperparameters (OPTIMIZED for Pendulum)
learning_rate  = 0.0003  # Higher for initial learning
gamma          = 0.99
lmbda          = 0.95
eps_clip       = 0.2
K_epoch        = 3       # Reduced to prevent overfitting
T_horizon      = 200     # Full episode as one trajectory

# Save/Load settings
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'ppo_continuous_pendulum'
SAVE_INTERVAL = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(3, 256)  # Larger network
        self.fc_mu = nn.Linear(256, 1)
        self.fc_std  = nn.Linear(256, 1)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Exponential decay: LR decay every episode, VERY slowly
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9999)
        self.optimization_step = 0

    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x)) + 0.001  # Add small epsilon
        return mu, std

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst), dtype=torch.float)
        r = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float)

        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # Calculate advantages with proper GAE
        with torch.no_grad():
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
        delta = delta.numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(np.array(advantage_lst), dtype=torch.float)

        # Normalize advantage for stability
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for i in range(K_epoch):
            mu, std = self.pi(s)
            dist = Normal(mu, std)
            log_prob = dist.log_prob(a)
            ratio = torch.exp(log_prob - prob_a)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)

            value_loss = F.smooth_l1_loss(self.v(s), td_target.detach())

            # Entropy bonus for exploration
            entropy = dist.entropy()
            entropy_loss = -0.01 * entropy

            loss = policy_loss + 0.5 * value_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.optimization_step += 1

            total_loss += loss.mean().item()
            total_policy_loss += policy_loss.mean().item()
            total_value_loss += value_loss.item()

        return total_loss / K_epoch, total_policy_loss / K_epoch, total_value_loss / K_epoch

def save_checkpoint(model, episode, best_score, avg_score, scores_window):
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'scheduler_state_dict': model.scheduler.state_dict(),
        'best_score': best_score,
        'avg_score': avg_score,
        'scores_window': list(scores_window),
        'optimization_step': model.optimization_step,
    }
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def save_best_model(model, episode, best_score):
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'best_score': best_score,
    }
    best_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_best.pth')
    torch.save(checkpoint, best_path)
    print(f"🏆 NEW BEST MODEL! Score: {best_score:.1f} → Saved to {best_path}")

def load_checkpoint(model):
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_episode = checkpoint['episode']
        best_score = checkpoint['best_score']
        scores_window = checkpoint.get('scores_window', [])
        scores_window = collections.deque(scores_window, maxlen=100)
        if 'optimization_step' in checkpoint:
            model.optimization_step = checkpoint['optimization_step']
        print(f"✅ Checkpoint loaded! Episode: {start_episode}, Best: {best_score:.1f}")
        return start_episode, best_score, scores_window
    else:
        print("🆕 No checkpoint found. Starting from scratch...")
        return 0, float('-inf'), collections.deque(maxlen=100)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/PPO_Continuous_Pendulum_{timestamp}')

    env = gym.make('Pendulum-v1')
    model = PPO()

    start_episode, best_score, scores_window = load_checkpoint(model)

    score = 0.0
    print_interval = 20

    print(f"\n{'='*60}")
    print(f"🚀 Starting PPO-Continuous Training (FIXED)")
    print(f"{'='*60}")
    print(f"Episodes: {start_episode} → 10000")
    print(f"Learning Rate: {learning_rate}")
    print(f"T_horizon: {T_horizon}")
    print(f"K_epoch: {K_epoch}")
    print(f"{'='*60}\n")

    for n_epi in range(start_episode, 10000):
        s, _ = env.reset()
        done = False
        episode_score = 0

        # Collect full episode trajectory
        for t in range(T_horizon):
            mu, std = model.pi(torch.from_numpy(s).float())
            dist = Normal(mu, std)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            s_prime, r, done, truncated, info = env.step([a.item()])

            # Better reward scaling: normalize to [-1, 0] range
            model.put_data((s, a.item(), r/16.0, s_prime, log_prob.item(), done))

            s = s_prime
            episode_score += r

            if done or truncated:
                break

        # Train after each episode
        avg_loss, policy_loss, value_loss = model.train_net()
        model.scheduler.step()  # Decay learning rate

        score += episode_score
        scores_window.append(episode_score)

        # Log to TensorBoard
        writer.add_scalar('Score/episode', episode_score, n_epi)
        writer.add_scalar('Score/average_100', np.mean(scores_window), n_epi)
        writer.add_scalar('Loss/total', avg_loss, n_epi)
        writer.add_scalar('OptimizationSteps', model.optimization_step, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            avg_100 = np.mean(scores_window)

            print(f"n_episode: {n_epi:5d}, score: {avg_score:7.1f}, avg_100: {avg_100:7.1f}, "
                  f"opt_step: {model.optimization_step:5d}, loss: {avg_loss:.4f}")

            save_checkpoint(model, n_epi, best_score, avg_score, scores_window)

            # Save best model
            if len(scores_window) >= 100:
                if avg_100 > best_score:
                    best_score = avg_100
                    save_best_model(model, n_epi, best_score)
            else:
                if avg_score > best_score:
                    best_score = avg_score
                    save_best_model(model, n_epi, best_score)

            score = 0.0

        # Check if solved (relaxed threshold for early stopping)
        if len(scores_window) == 100 and np.mean(scores_window) >= -250.0:
            print(f"\n🎉 Environment SOLVED in {n_epi} episodes! Avg: {np.mean(scores_window):.2f}")
            print(f"🛑 Early stopping to prevent overfitting!")
            save_best_model(model, n_epi, np.mean(scores_window))
            break

    final_path = save_checkpoint(model, n_epi, best_score, score/print_interval, scores_window)
    print(f"\n{'='*60}")
    print(f"✅ Training completed! Best Score: {best_score:.1f}")
    print(f"💾 Saved to: {final_path}")
    print(f"{'='*60}\n")

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
