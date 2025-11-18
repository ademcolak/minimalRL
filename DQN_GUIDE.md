# 🎮 DQN CartPole - Full Version Guide

Enhanced DQN implementation with model persistence, TensorBoard logging, and advanced features.

---

## 📋 **Features**

✅ **Model Persistence**
- Automatic checkpoint saving every 20 episodes
- Best model tracking
- Resume training from checkpoint
- Optimizer state saving

✅ **TensorBoard Logging**
- Real-time training metrics
- Loss, Q-values, scores visualization
- Epsilon decay tracking

✅ **Replay Buffer Management**
- Optional buffer saving/loading
- 50K experience capacity
- Efficient numpy-based tensor conversion

✅ **Advanced Statistics**
- 100-episode rolling average
- Automatic "solved" detection (avg >= 195)
- Detailed progress reporting

✅ **Production Ready**
- Clean code structure
- Comprehensive error handling
- Easy testing interface

---

## 🚀 **Quick Start**

### **1. Train the Model**

```bash
python3 dqn.py
```

**First Run:**
```
🆕 No checkpoint found. Starting from scratch...
============================================================
🚀 Starting DQN Training on CartPole-v1
============================================================
Episodes: 0 → 2000
Learning Rate: 0.0005
Gamma: 0.98
Buffer Limit: 50000
Batch Size: 32
TensorBoard: runs/DQN_CartPole_20250118_143052
============================================================

n_episode :   20, score : 18.5, avg_100: 18.5, n_buffer :  370, eps : 7.0%, loss: 0.0234
...
```

**Resume Training:**
```
📂 Loading checkpoint: checkpoints/dqn_cartpole_latest.pth
✅ Checkpoint loaded!
   Episode: 820
   Best Score: 195.3

n_episode :  840, score : 197.1, avg_100: 196.2, ...  ← Continues!
```

---

### **2. Test the Model**

#### **Basic Test (10 episodes):**
```bash
python3 test_dqn.py
```

Output:
```
📂 Loading model: checkpoints/dqn_cartpole_best.pth
✅ Model loaded!
   Episode: 820
   Best Score: 195.3

============================================================
🎮 Running 10 test episodes...
============================================================

Episode  1: Score =  198.0, Steps = 198
Episode  2: Score =  200.0, Steps = 200
Episode  3: Score =  195.0, Steps = 195
...

============================================================
📊 Test Results:
============================================================
Average Score: 197.50 ± 2.13
Min Score:     195.0
Max Score:     200.0
Success Rate:  100.0%
============================================================

✅ Model is performing well! (avg >= 195)
```

#### **Custom Options:**
```bash
# Test specific model
python3 test_dqn.py --model checkpoints/dqn_cartpole_latest.pth

# Test 50 episodes
python3 test_dqn.py --episodes 50

# Render environment (requires display)
python3 test_dqn.py --render

# Compare best vs latest model
python3 test_dqn.py --compare
```

---

### **3. View TensorBoard**

```bash
# Start TensorBoard server
chmod +x view_tensorboard.sh
./view_tensorboard.sh

# Or manually:
tensorboard --logdir=runs --port=6006
```

Then open browser: **http://localhost:6006**

**Metrics Available:**
- `Loss/train` - Training loss over time
- `Q_value/average` - Average Q-values
- `Score/episode` - Score per episode
- `Score/average_100` - 100-episode rolling average
- `Epsilon` - Exploration rate decay
- `Buffer_size` - Replay buffer growth

---

## 📁 **File Structure**

```
minimalRL/
├── dqn.py                  # Main training script (FULL VERSION)
├── test_dqn.py             # Testing script
├── DQN_GUIDE.md            # This guide
├── view_tensorboard.sh     # TensorBoard launcher
├── checkpoints/            # Saved models (auto-created)
│   ├── dqn_cartpole_latest.pth    # Latest checkpoint
│   ├── dqn_cartpole_best.pth      # Best model
│   └── dqn_cartpole_buffer.pkl    # Replay buffer (optional)
└── runs/                   # TensorBoard logs (auto-created)
    └── DQN_CartPole_20250118_143052/
```

---

## ⚙️ **Hyperparameters**

```python
learning_rate = 0.0005   # Adam optimizer learning rate
gamma         = 0.98     # Discount factor
buffer_limit  = 50000    # Replay buffer capacity
batch_size    = 32       # Mini-batch size
```

**Adjusting Episodes:**
```python
# In dqn.py, line 217
for n_epi in range(start_episode, 2000):  # Change 2000 to desired value
```

---

## 🎯 **Training Workflow**

### **1. Initial Training**
```bash
python3 dqn.py
# Runs episodes 0 → 2000
# Saves checkpoint every 20 episodes
# Stops early if solved (avg >= 195)
```

### **2. Continue Training**
```bash
python3 dqn.py
# Loads latest checkpoint
# Continues from last episode
# Updates best model if improved
```

### **3. Test Performance**
```bash
python3 test_dqn.py --episodes 100
# Tests best model over 100 episodes
# Shows detailed statistics
```

### **4. Analyze with TensorBoard**
```bash
./view_tensorboard.sh
# View training curves
# Compare different runs
```

---

## 💾 **Checkpoint Format**

```python
checkpoint = {
    'episode': 820,                      # Episode number
    'model_state_dict': {...},           # Q-network weights
    'target_state_dict': {...},          # Target network weights
    'optimizer_state_dict': {...},       # Adam optimizer state
    'best_score': 195.3,                 # Best average score
    'avg_score': 197.1                   # Current average score
}
```

**File Size:**
- `dqn_cartpole_latest.pth`: ~146 KB
- `dqn_cartpole_best.pth`: ~146 KB
- `dqn_cartpole_buffer.pkl`: ~5 MB (optional)

---

## 📊 **Understanding Output**

```
n_episode :  500, score : 185.2, avg_100: 178.5, n_buffer : 50000, eps : 1.0%, loss: 0.0056
     ↑           ↑                ↑                ↑                ↑            ↑
  Episode    Last 20 avg    Last 100 avg    Buffer size    Exploration    Train loss
```

**Key Metrics:**
- `score`: Average score over last 20 episodes
- `avg_100`: Average score over last 100 episodes (solve criteria)
- `n_buffer`: Number of experiences in replay buffer
- `eps`: Current epsilon (exploration rate)
- `loss`: Training loss (should decrease over time)

---

## 🎓 **Tips**

### **Training Too Slow?**
```python
# Reduce episodes
for n_epi in range(start_episode, 1000):  # Instead of 2000

# Reduce buffer saving frequency
save_buffer=False  # In save_checkpoint() call
```

### **Training Too Fast?**
```python
# Increase episodes
for n_epi in range(start_episode, 5000):

# Stricter solve criteria
if np.mean(scores_window) >= 200.0:  # Instead of 195.0
```

### **Want to Start Fresh?**
```bash
# Delete checkpoints
rm -rf checkpoints/

# Delete TensorBoard logs
rm -rf runs/
```

### **Multiple Experiments?**
```python
# Change model name in dqn.py
MODEL_NAME = 'dqn_cartpole_experiment2'
```

---

## 🐛 **Troubleshooting**

### **"No checkpoint found"**
✅ Normal on first run. Model will train from scratch.

### **"Model not found" when testing**
```bash
# Train first
python3 dqn.py

# Then test
python3 test_dqn.py
```

### **TensorBoard not showing graphs**
```bash
# Check if runs/ directory exists
ls runs/

# Refresh browser
# Wait a few seconds for data to load
```

### **Memory issues with replay buffer**
```python
# In dqn.py, reduce buffer size
buffer_limit = 10000  # Instead of 50000
```

---

## 📈 **Expected Performance**

| Episode | Score | Status |
|---------|-------|--------|
| 0-100 | 10-50 | 🔴 Learning basics |
| 100-300 | 50-120 | 🟡 Improving |
| 300-600 | 120-180 | 🟢 Good progress |
| 600-1000 | 180-195 | 🟢 Almost solved |
| 1000+ | 195+ | ✅ **SOLVED** |

**Typical solve time:** 800-1200 episodes

---

## 🔬 **Advanced Usage**

### **Load Specific Checkpoint**
```python
# In test_dqn.py
test_model('checkpoints/dqn_cartpole_latest.pth', num_episodes=100)
```

### **Save Replay Buffer**
```python
# In dqn.py, line 259
save_checkpoint(..., save_buffer=True)  # Enable buffer saving
```

### **Load Replay Buffer**
```python
# In dqn.py, line 200
start_episode, best_score = load_checkpoint(..., load_buffer=True)
```

### **Custom Network Architecture**
```python
# In dqn.py, modify Qnet class
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256)   # Larger network
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
```

---

## 📚 **Next Steps**

1. ✅ **Train the model** → Run until "SOLVED"
2. ✅ **Test performance** → Verify avg >= 195
3. ✅ **Analyze metrics** → Check TensorBoard
4. ✅ **Experiment** → Adjust hyperparameters
5. ✅ **Compare runs** → Use TensorBoard to compare

---

## 🤝 **Contributing**

Improvements welcome! This is the full version with:
- ✅ Model persistence
- ✅ TensorBoard logging
- ✅ Replay buffer saving
- ✅ Best model tracking
- ✅ Comprehensive testing

Enjoy training! 🚀
