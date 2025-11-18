import gymnasium as gym
import collections
import random
import numpy as np
import os
import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

# Save/Load settings
SAVE_DIR = 'checkpoints'
MODEL_NAME = 'dqn_cartpole'
SAVE_INTERVAL = 20  # Save every N episodes

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.FloatTensor(np.array(s_lst)), torch.tensor(np.array(a_lst)), \
               torch.tensor(np.array(r_lst)), torch.FloatTensor(np.array(s_prime_lst)), \
               torch.tensor(np.array(done_mask_lst))

    def size(self):
        return len(self.buffer)

    def save(self, path):
        """Save replay buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"💾 Replay buffer saved: {path}")

    def load(self, path):
        """Load replay buffer from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                buffer_list = pickle.load(f)
                self.buffer = collections.deque(buffer_list, maxlen=buffer_limit)
            print(f"📂 Replay buffer loaded: {path} ({len(self.buffer)} experiences)")
            return True
        return False

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

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else :
            return out.argmax().item()

def train(q, q_target, memory, optimizer, writer=None, step=0):
    total_loss = 0.0
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Log to TensorBoard
    if writer is not None:
        avg_loss = total_loss / 10
        avg_q = q_out.mean().item()
        writer.add_scalar('Loss/train', avg_loss, step)
        writer.add_scalar('Q_value/average', avg_q, step)

    return total_loss / 10

def save_checkpoint(q, q_target, optimizer, memory, episode, best_score, avg_score, save_buffer=False):
    """Save model checkpoint"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'model_state_dict': q.state_dict(),
        'target_state_dict': q_target.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
        'avg_score': avg_score,
    }

    # Save latest checkpoint
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')
    torch.save(checkpoint, checkpoint_path)

    # Save replay buffer separately (optional, can be large)
    if save_buffer:
        buffer_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_buffer.pkl')
        memory.save(buffer_path)

    return checkpoint_path

def save_best_model(q, q_target, optimizer, episode, best_score):
    """Save best model separately"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    checkpoint = {
        'episode': episode,
        'model_state_dict': q.state_dict(),
        'target_state_dict': q_target.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_score': best_score,
    }

    best_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_best.pth')
    torch.save(checkpoint, best_path)
    print(f"🏆 NEW BEST MODEL! Score: {best_score:.1f} → Saved to {best_path}")

def load_checkpoint(q, q_target, optimizer, memory, load_buffer=False):
    """Load model checkpoint if exists"""
    checkpoint_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_latest.pth')

    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        q.load_state_dict(checkpoint['model_state_dict'])
        q_target.load_state_dict(checkpoint['target_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_episode = checkpoint['episode']
        best_score = checkpoint['best_score']

        print(f"✅ Checkpoint loaded!")
        print(f"   Episode: {start_episode}")
        print(f"   Best Score: {best_score:.1f}")

        # Load replay buffer if requested
        if load_buffer:
            buffer_path = os.path.join(SAVE_DIR, f'{MODEL_NAME}_buffer.pkl')
            memory.load(buffer_path)

        return start_episode, best_score
    else:
        print("🆕 No checkpoint found. Starting from scratch...")
        return 0, 0.0

def main(render=False):
    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/DQN_CartPole_{timestamp}')

    # Create environment with optional rendering
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
        print("🎮 Rendering enabled - you will see the agent play!")
    else:
        env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # Load checkpoint if exists
    start_episode, best_score = load_checkpoint(q, q_target, optimizer, memory, load_buffer=False)

    print_interval = 5  # Changed from 20 to 5 for faster feedback
    score = 0.0  # Always reset score when starting/resuming training
    scores_window = collections.deque(maxlen=200)  # Last 100 scores

    print(f"\n{'='*60}")
    print(f"🚀 Starting DQN Training on CartPole-v1")
    print(f"{'='*60}")
    print(f"Episodes: {start_episode} → 2000")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Buffer Limit: {buffer_limit}")
    print(f"Batch Size: {batch_size}")
    print(f"TensorBoard: runs/DQN_CartPole_{timestamp}")
    print(f"{'='*60}\n")
    print("Training started... (output every 5 episodes)")

    for n_epi in range(start_episode, 2000):  # Your reduced episode count
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False
        episode_score = 0

        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            episode_score += r
            # Episode ends if either done OR truncated (time limit reached)
            if done or truncated:
                break

        score += episode_score
        scores_window.append(episode_score)

        # Train if enough samples
        if memory.size()>2000:
            avg_loss = train(q, q_target, memory, optimizer, writer, n_epi)
        else:
            avg_loss = 0.0

        # Logging to TensorBoard
        writer.add_scalar('Score/episode', episode_score, n_epi)
        writer.add_scalar('Score/average_100', np.mean(scores_window), n_epi)
        writer.add_scalar('Epsilon', epsilon, n_epi)
        writer.add_scalar('Buffer_size', memory.size(), n_epi)

        # Print and save periodically
        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            avg_score = score/print_interval
            avg_100 = np.mean(scores_window)

            print("n_episode :{:5d}, score : {:.1f}, avg_100: {:.1f}, n_buffer : {:5d}, eps : {:.1f}%, loss: {:.4f}".format(
                n_epi, avg_score, avg_100, memory.size(), epsilon*100, avg_loss))

            # Save checkpoint
            save_checkpoint(q, q_target, optimizer, memory, n_epi, best_score, avg_score, save_buffer=False)

            # Save best model
            if avg_score > best_score:
                best_score = avg_score
                save_best_model(q, q_target, optimizer, n_epi, best_score)

            score = 0.0

        # Check if solved (average score >= 195 over last 100 episodes)
        if len(scores_window) == 100 and np.mean(scores_window) >= 195.0:
            print(f"\n🎉 Environment SOLVED in {n_epi} episodes!")
            print(f"🎉 Average Score: {np.mean(scores_window):.2f}")
            save_best_model(q, q_target, optimizer, n_epi, np.mean(scores_window))
            break

    # Final save
    final_path = save_checkpoint(q, q_target, optimizer, memory, n_epi, best_score, score/print_interval, save_buffer=True)
    print(f"\n{'='*60}")
    print(f"✅ Training completed!")
    print(f"📊 Best Score: {best_score:.1f}")
    print(f"💾 Model saved to: {final_path}")
    print(f"📈 TensorBoard logs: runs/DQN_CartPole_{timestamp}")
    print(f"{'='*60}\n")

    env.close()
    writer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train DQN on CartPole-v1')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering to watch training (slower)')

    args = parser.parse_args()
    main(render=args.render)
