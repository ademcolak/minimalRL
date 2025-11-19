import gymnasium as gym
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

# Import from ppo-continuous
exec(open('ppo-continuous-fixed.py').read().split('def main()')[0])

# Test the model
env = gym.make('Pendulum-v1')
model = PPO()

print("=" * 60)
print("🔍 DEBUG: Testing PPO Continuous")
print("=" * 60)

# Run a few steps and check values
s, _ = env.reset()
rollout = []

print(f"\n1️⃣ Initial state shape: {s.shape}")
print(f"   Initial state values: {s}")

for step in range(10):
    # Get action
    mu, std = model.pi(torch.from_numpy(s).float())
    print(f"\n2️⃣ Step {step}:")
    print(f"   mu: {mu.item():.4f}, std: {std.item():.4f}")

    from torch.distributions import Normal
    dist = Normal(mu, std)
    a = dist.sample()
    log_prob = dist.log_prob(a)

    print(f"   action: {a.item():.4f}, log_prob: {log_prob.item():.4f}")

    s_prime, r, done, truncated, info = env.step([a.item()])

    print(f"   reward: {r:.4f}, r/10.0: {r/10.0:.4f}")

    rollout.append((s, a.item(), r/10.0, s_prime, log_prob.item(), done))

    if len(rollout) == 5:
        model.put_data(rollout)
        rollout = []

        if len(model.data) >= 2:
            print(f"\n3️⃣ Training with {len(model.data)} rollouts...")
            s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch, old_log_prob_batch = model.make_batch()

            print(f"   Batch shapes:")
            print(f"   - s: {s_batch.shape}")
            print(f"   - a: {a_batch.shape}")
            print(f"   - r: {r_batch.shape}")
            print(f"   - r min/max: {r_batch.min().item():.4f} / {r_batch.max().item():.4f}")

            td_target, advantage = model.calc_advantage(s_batch, a_batch, r_batch, s_prime_batch, done_mask_batch, old_log_prob_batch)

            print(f"   - td_target min/max: {td_target.min().item():.4f} / {td_target.max().item():.4f}")
            print(f"   - advantage mean/std: {advantage.mean().item():.4f} / {advantage.std().item():.4f}")

            avg_loss, policy_loss, value_loss = model.train_net()

            print(f"   - avg_loss: {avg_loss:.6f}")
            print(f"   - policy_loss: {policy_loss:.6f}")
            print(f"   - value_loss: {value_loss:.6f}")

            break

    s = s_prime
    if done or truncated:
        break

print(f"\n✅ Debug complete!")
env.close()
