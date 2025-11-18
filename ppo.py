import gymnasium as gym
import numpy as np
import os
import pickle
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

# Save/Load settings
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'ppo_cartpole'

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(np.array(s_lst), dtype=torch.float), \
                                          torch.tensor(np.array(a_lst)), \
                                          torch.tensor(np.array(r_lst)), \
                                          torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
                                          torch.tensor(np.array(done_lst), dtype=torch.float), \
                                          torch.tensor(np.array(prob_a_lst))
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, writer=None, step=0):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        total_loss = 0.0
        total_pi_loss = 0.0
        total_v_loss = 0.0

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

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            pi_loss = -torch.min(surr1, surr2)
            v_loss = F.smooth_l1_loss(self.v(s), td_target.detach())
            loss = pi_loss + v_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            total_loss += loss.mean().item()
            total_pi_loss += pi_loss.mean().item()
            total_v_loss += v_loss.mean().item()

        # Log to TensorBoard
        if writer is not None:
            avg_loss = total_loss / K_epoch
            avg_pi_loss = total_pi_loss / K_epoch
            avg_v_loss = total_v_loss / K_epoch
            avg_v = self.v(s).mean().item()

            writer.add_scalar('Loss/total', avg_loss, step)
            writer.add_scalar('Loss/policy', avg_pi_loss, step)
            writer.add_scalar('Loss/value', avg_v_loss, step)
            writer.add_scalar('Value/average', avg_v, step)
            writer.add_scalar('Advantage/mean', advantage.mean().item(), step)
            writer.add_scalar('Advantage/std', advantage.std().item(), step)

        return total_loss / K_epoch

def save_checkpoint(model, episode, best_score, avg_score):
    """Save model checkpoint"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'best_score': best_score,
        'avg_score': avg_score,
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path

def save_best_model(model, episode, best_score):
    """Save best model separately"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
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

        start_episode = checkpoint['episode']
        best_score = checkpoint['best_score']

        print(f"✅ Checkpoint loaded!")
        print(f"   Episode: {start_episode}")
        print(f"   Best Score: {best_score:.1f}")

        return start_episode, best_score
    else:
        print("🆕 No checkpoint found. Starting from scratch...")
        return 0, 0.0

def main():
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/PPO_CartPole_{timestamp}')

    env = gym.make('CartPole-v1')
    model = PPO()

    # Load checkpoint if exists
    start_episode, best_score = load_checkpoint(model)

    print_interval = 5  # Reduced from 20 for better feedback
    score = 0.0
    scores_window = deque(maxlen=100)  # Last 100 scores

    print(f"\n{'='*60}")
    print(f"🚀 Starting PPO Training on CartPole-v1")
    print(f"{'='*60}")
    print(f"Episodes: {start_episode} → 2000")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Lambda (GAE): {lmbda}")
    print(f"Epsilon Clip: {eps_clip}")
    print(f"K Epochs: {K_epoch}")
    print(f"T Horizon: {T_horizon}")
    print(f"Print Interval: {print_interval}")
    print(f"TensorBoard: runs/PPO_CartPole_{timestamp}")
    print(f"{'='*60}\n")

    for n_epi in range(start_episode, 2000):
        s, _ = env.reset()
        done = False
        truncated = False
        episode_score = 0

        while not (done or truncated):  # FIX: Check both done and truncated
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, info = env.step(a)

                # FIX: Use done OR truncated for episode termination
                episode_done = done or truncated
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), episode_done))
                s = s_prime

                episode_score += r
                if done or truncated:  # FIX: Check both
                    break

            avg_loss = model.train_net(writer, n_epi)

        score += episode_score
        scores_window.append(episode_score)

        # Logging to TensorBoard
        writer.add_scalar('Score/episode', episode_score, n_epi)
        writer.add_scalar('Score/average_100', np.mean(scores_window), n_epi)

        if n_epi%print_interval==0 and n_epi!=0:
            avg_score = score/print_interval
            avg_100 = np.mean(scores_window)

            print("n_episode :{:5d}, score : {:.1f}, avg_100: {:.1f}, loss: {:.4f}".format(
                n_epi, avg_score, avg_100, avg_loss))

            # Save checkpoint
            save_checkpoint(model, n_epi, best_score, avg_score)

            # Save best model
            if avg_score > best_score:
                best_score = avg_score
                save_best_model(model, n_epi, best_score)

            score = 0.0

        # Check if solved (average score >= 195 over last 100 episodes)
        if len(scores_window) == 100 and np.mean(scores_window) >= 195.0:
            print(f"\n🎉 Environment SOLVED in {n_epi} episodes!")
            print(f"🎉 Average Score: {np.mean(scores_window):.2f}")
            save_best_model(model, n_epi, np.mean(scores_window))
            break

    # Final save
    final_path = save_checkpoint(model, n_epi, best_score, score/print_interval if score > 0 else best_score)
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"📊 Best Score: {best_score:.1f}")
    print(f"💾 Model saved to: {final_path}")
    print(f"📈 TensorBoard logs: runs/PPO_CartPole_{timestamp}")
    print(f"{'='*60}\n")

    env.close()
    writer.close()

if __name__ == '__main__':
    main()
