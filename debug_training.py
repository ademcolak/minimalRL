import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

# Minimal test setup
learning_rate = 0.0002
gamma = 0.99
lmbda = 0.95
eps_clip = 0.2
K_epoch = 3
rollout_len = 64
buffer_size = 5

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(3,128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std = nn.Linear(128,1)
        self.fc_v = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0

    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for rollout in self.data:
            for transition in rollout:
                s, a, r, s_prime, prob_a, done = transition
                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])

        self.data = []
        s = torch.tensor(np.array(s_lst), dtype=torch.float)
        a = torch.tensor(np.array(a_lst), dtype=torch.float)
        r = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float)
        return s, a, r, s_prime, done_mask, prob_a

    def calc_advantage(self, s, a, r, s_prime, done_mask, old_log_prob):
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
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return td_target, advantage

    def train_net(self):
        print(f"🔍 train_net called, data length: {len(self.data)}")

        if len(self.data) >= buffer_size:
            print(f"✅ Condition met! Training...")
            s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
            print(f"   Batch created: s={s.shape}, a={a.shape}, r range=[{r.min():.3f}, {r.max():.3f}]")

            td_target, advantage = self.calc_advantage(s, a, r, s_prime, done_mask, old_log_prob)
            print(f"   Advantage: mean={advantage.mean():.6f}, std={advantage.std():.6f}")

            total_loss = 0.0
            for i in range(K_epoch):
                mu, std = self.pi(s)
                dist = Normal(mu, std)
                log_prob = dist.log_prob(a)
                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                policy_loss = -torch.min(surr1, surr2)
                value_loss = F.smooth_l1_loss(self.v(s), td_target)
                entropy = dist.entropy()
                entropy_loss = -0.01 * entropy
                loss = policy_loss + 0.5 * value_loss + entropy_loss

                print(f"   Epoch {i}: loss={loss.mean().item():.6f}, p_loss={policy_loss.mean().item():.6f}, v_loss={value_loss.item():.6f}")

                self.optimizer.zero_grad()
                loss.mean().backward()

                # Check gradients
                grad_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                print(f"     Grad norm: {grad_norm:.6f}")

                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.optimization_step += 1
                total_loss += loss.mean().item()

            return total_loss / K_epoch, 0.0, 0.0
        else:
            print(f"❌ Condition NOT met, returning 0s")
            return 0.0, 0.0, 0.0

# Test
env = gym.make('Pendulum-v1')
model = PPO()

s, _ = env.reset()
rollout = []
episode = 0

print("="*60)
print("Starting debug training...")
print("="*60)

for step in range(500):
    mu, std = model.pi(torch.from_numpy(s).float())
    dist = Normal(mu, std)
    a = dist.sample()
    log_prob = dist.log_prob(a)
    s_prime, r, done, truncated, info = env.step([a.item()])

    rollout.append((s, a.item(), r/10.0, s_prime, log_prob.item(), done))

    if len(rollout) == rollout_len:
        model.put_data(rollout)
        rollout = []
        print(f"\nStep {step}: Added rollout, total rollouts: {len(model.data)}")

    s = s_prime

    if done or truncated:
        s, _ = env.reset()
        episode += 1

    # Try training
    if len(model.data) >= buffer_size:
        print(f"\n{'='*60}")
        print(f"Training at step {step}, episode {episode}")
        avg_loss, _, _ = model.train_net()
        print(f"Returned avg_loss: {avg_loss:.6f}")
        print(f"{'='*60}\n")
        break

env.close()
