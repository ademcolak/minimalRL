import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import collections
import numpy as np
import os
from datetime import datetime

#Hyperparameters (Improved for better stability)
learning_rate     = 0.0003  # Initial learning rate (will decay)
gamma             = 0.98
lmbda             = 0.95
eps_clip          = 0.2     # Increased from 0.1 for more exploration
K_epoch           = 3       # Reduced to prevent overfitting on same data
T_horizon         = 50      # Increased from 20 for longer trajectories
entropy_coef_start = 0.1    # Start with high exploration
entropy_coef_end   = 0.01   # Decay to low exploration
max_episodes      = 10000   # For entropy decay calculation

# Save/Load settings
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'ppo_cartpole'
SAVE_INTERVAL = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Learning rate scheduler - gradual decay (slower than before)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
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
        a = torch.tensor(np.array(a_lst))
        r = torch.tensor(np.array(r_lst))
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self, episode=0):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # Adaptive entropy coefficient - decay over time
        progress = min(episode / max_episodes, 1.0)
        entropy_coef = entropy_coef_start - (entropy_coef_start - entropy_coef_end) * progress

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # Normalize advantage for stability
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            # Add epsilon to prevent log(0)
            ratio = torch.exp(torch.log(pi_a + 1e-8) - torch.log(prob_a + 1e-8))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.smooth_l1_loss(self.v(s), td_target.detach())

            # Entropy bonus for exploration
            entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1, keepdim=True)
            entropy_bonus = entropy_coef * entropy

            # Combined loss with coefficients
            loss = policy_loss + 0.5 * value_loss - entropy_bonus

            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.mean().item()
            total_policy_loss += policy_loss.mean().item()
            total_value_loss += value_loss.mean().item()

        return total_loss / K_epoch, total_policy_loss / K_epoch, total_value_loss / K_epoch, entropy_coef
        
def save_checkpoint(model, episode, best_score, avg_score, scores_window, episodes_since_improvement=0):
    """Save model checkpoint"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'scheduler_state_dict': model.scheduler.state_dict(),
        'best_score': best_score,
        'avg_score': avg_score,
        'scores_window': list(scores_window),  # Save last 100 scores
        'episodes_since_improvement': episodes_since_improvement,
    }

    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

def save_best_model(model, episode, best_score):
    """Save best model separately"""
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
    """Load model checkpoint if exists"""
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')

    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler if available (for backward compatibility)
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_episode = checkpoint['episode']
        best_score = checkpoint['best_score']

        # Load scores window if available (for backward compatibility)
        scores_window = checkpoint.get('scores_window', [])
        scores_window = collections.deque(scores_window, maxlen=100)

        # Load episodes_since_improvement if available (for backward compatibility)
        episodes_since_improvement = checkpoint.get('episodes_since_improvement', 0)

        print(f"✅ Checkpoint loaded!")
        print(f"   Episode: {start_episode}")
        print(f"   Best Score: {best_score:.1f}")
        print(f"   Scores Window Size: {len(scores_window)}")
        print(f"   Episodes Since Improvement: {episodes_since_improvement}")

        return start_episode, best_score, scores_window, episodes_since_improvement
    else:
        print("🆕 No checkpoint found. Starting from scratch...")
        return 0, 0.0, collections.deque(maxlen=100), 0

def main():
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/PPO_CartPole_{timestamp}')

    env = gym.make('CartPole-v1')
    model = PPO()

    # Load checkpoint if exists
    start_episode, best_score, scores_window, episodes_since_improvement = load_checkpoint(model)

    score = 0.0
    print_interval = 5  # Changed from 20 for faster feedback

    # Early stopping parameters
    patience = 3000  # Stop if no improvement for 3000 episodes
    degradation_check_window = 500  # Check degradation over last 500 episodes
    degradation_threshold = 0.5  # Alert if sustained performance drops to 50% of best
    last_degradation_warning = -1000  # Track when we last warned (avoid spam)

    print(f"\n{'='*60}")
    print(f"🚀 Starting PPO Training on CartPole-v1")
    print(f"{'='*60}")
    print(f"Episodes: {start_episode} → 10000")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Lambda: {lmbda}")
    print(f"Epsilon Clip: {eps_clip}")
    print(f"K Epochs: {K_epoch}")
    print(f"T Horizon: {T_horizon}")
    print(f"TensorBoard: runs/PPO_CartPole_{timestamp}")
    print(f"{'='*60}\n")
    print("Training started... (output every 5 episodes)")

    for n_epi in range(start_episode, 10000):  # Train up to 10000 episodes total
        s, _ = env.reset()
        done = False
        episode_score = 0

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                episode_score += r
                if done or truncated:
                    break

            avg_loss, policy_loss, value_loss, entropy_coef = model.train_net(episode=n_epi)

        score += episode_score
        scores_window.append(episode_score)

        # Step learning rate scheduler every episode
        model.scheduler.step()

        # Log to TensorBoard
        writer.add_scalar('Score/episode', episode_score, n_epi)
        writer.add_scalar('Score/average_100', np.mean(scores_window), n_epi)
        writer.add_scalar('Loss/total', avg_loss, n_epi)
        writer.add_scalar('Loss/policy', policy_loss, n_epi)
        writer.add_scalar('Loss/value', value_loss, n_epi)
        writer.add_scalar('LearningRate', model.optimizer.param_groups[0]['lr'], n_epi)
        writer.add_scalar('EntropyCoef', entropy_coef, n_epi)

        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            avg_100 = np.mean(scores_window)

            print("n_episode :{:5d}, score : {:.1f}, avg_100: {:.1f}, loss: {:.4f}".format(
                n_epi, avg_score, avg_100, avg_loss))

            # Save checkpoint
            save_checkpoint(model, n_epi, best_score, avg_score, scores_window, episodes_since_improvement)

            # Save best model and check for improvement
            # Use avg_100 for more stable comparison (only after 100 episodes)
            if len(scores_window) >= 100:
                if avg_100 > best_score:
                    best_score = avg_100
                    episodes_since_improvement = 0
                    save_best_model(model, n_epi, best_score)
                else:
                    episodes_since_improvement += print_interval
            else:
                # Before 100 episodes, use avg_score
                if avg_score > best_score:
                    best_score = avg_score
                    episodes_since_improvement = 0
                    save_best_model(model, n_epi, best_score)
                else:
                    episodes_since_improvement += print_interval

            # Early stopping check
            if episodes_since_improvement >= patience and best_score > 0:
                print(f"\n⏸️  Early stopping triggered!")
                print(f"   No improvement for {patience} episodes")
                print(f"   Best score: {best_score:.1f}")
                print(f"   Current avg_100: {avg_100:.1f}")
                break

            # Performance degradation warning - only if sustained and severe
            # Check if we have enough data and if we've been stuck for a while
            if (best_score > 250 and  # Only check if we had good performance (raised threshold)
                episodes_since_improvement >= degradation_check_window and  # Been a while (500 eps)
                avg_100 < best_score * degradation_threshold and  # Significant drop (50%)
                n_epi - last_degradation_warning >= 500):  # Don't spam warnings
                print(f"\n⚠️  WARNING: Sustained performance degradation detected!")
                print(f"   Current avg_100: {avg_100:.1f} < {best_score * degradation_threshold:.1f} (50% of best)")
                print(f"   Best score: {best_score:.1f}")
                print(f"   No improvement for {episodes_since_improvement} episodes")
                print(f"   💡 Recommendation: Consider stopping and using best model")
                last_degradation_warning = n_epi

            score = 0.0

        # Check if solved (higher threshold for better performance)
        if len(scores_window) == 100 and np.mean(scores_window) >= 475.0:
            print(f"\n🎉 Environment SOLVED in {n_epi} episodes!")
            print(f"🎉 Average Score: {np.mean(scores_window):.2f}")
            save_best_model(model, n_epi, np.mean(scores_window))
            break

    # Final save
    final_path = save_checkpoint(model, n_epi, best_score, score/print_interval, scores_window, episodes_since_improvement)
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"📊 Best Score: {best_score:.1f}")
    print(f"📉 Episodes Since Improvement: {episodes_since_improvement}")
    print(f"💾 Model saved to: {final_path}")
    print(f"📈 TensorBoard logs: runs/PPO_CartPole_{timestamp}")
    print(f"{'='*60}\n")
    print(f"💡 Tip: Use 'python3 test_ppo.py --compare' to compare best vs latest models")

    env.close()
    writer.close()

if __name__ == '__main__':
    main()