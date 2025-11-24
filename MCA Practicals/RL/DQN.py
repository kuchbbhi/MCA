import gymnasium as gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import random 
import numpy as np 
from collections import deque 
import matplotlib.pyplot as plt 
 
# ------------------------- 
# 1. Define Q-Network (The Deep Neural Network)
# ------------------------- 
# The network takes the state (4 values for CartPole) and outputs Q-values for each action (2 actions)
class DQN(nn.Module): 
    def __init__(self, state_dim, action_dim): 
        super(DQN, self).__init__() 
        self.net = nn.Sequential( 
            nn.Linear(state_dim, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, action_dim) 
        ) 
 
    def forward(self, x): 
        return self.net(x) 
 
# ------------------------- 
# 2. Experience Replay Buffer (Breaks correlation between samples)
# ------------------------- 
class ReplayBuffer: 
    def __init__(self, capacity=10000): 
        self.buffer = deque(maxlen=capacity) 
 
    def push(self, state, action, reward, next_state, done): 
        """Saves a transition (S, A, R, S', D) to the buffer."""
        self.buffer.append((state, action, reward, next_state, done)) 
 
    def sample(self, batch_size): 
        """Samples a random batch of transitions and converts them to PyTorch tensors."""
        batch = random.sample(self.buffer, batch_size) 
        # Unzip the batch and convert to numpy arrays first
        state, action, reward, next_state, done = map(np.array, zip(*batch)) 
        
        # Convert to PyTorch Tensors
        return ( 
            torch.FloatTensor(state), 
            torch.LongTensor(action),
            torch.FloatTensor(reward), 
            torch.FloatTensor(next_state), 
            torch.FloatTensor(done) 
        ) 
 
    def __len__(self): 
        return len(self.buffer) 
 
# ------------------------- 
# 3. DQN Training Loop (The main algorithm)
# ------------------------- 
def train_dqn(env_name="CartPole-v1", episodes=500): 
    # Environment Setup
    env = gym.make(env_name) 
    state_dim = env.observation_space.shape[0] # 4 for CartPole
    action_dim = env.action_space.n           # 2 for CartPole (Left or Right)
 
    # Network Initialization
    q_net = DQN(state_dim, action_dim) 
    target_net = DQN(state_dim, action_dim) 
    target_net.load_state_dict(q_net.state_dict()) # Target Network starts identical to Q-Network
 
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3) 
    replay_buffer = ReplayBuffer(10000) 
 
    # Hyperparameters
    gamma = 0.99 
    epsilon = 1.0 
    epsilon_min = 0.05 
    epsilon_decay = 0.995 
    batch_size = 64 
    target_update = 20 # How often to update the Target Network
 
    rewards_history = [] 
 
    for ep in range(episodes): 
        state, _ = env.reset() 
        total_reward = 0 
 
        for t in range(500): 
            # 1. ε-greedy action selection 
            if random.random() < epsilon: 
                action = env.action_space.sample() 
            else: 
                # Exploit: choose the action with the max Q-value from the Q-net
                with torch.no_grad(): 
                    # Convert state to tensor, pass through net, find max Q-value index, get Python integer
                    action = q_net(torch.FloatTensor(state)).argmax().item() 
 
            # 2. Interact with environment
            next_state, reward, done, truncated, _ = env.step(action) 
            
            # 3. Store transition in buffer
            # Note: reward is scaled by the problem implicitly (it's +1 per step alive)
            replay_buffer.push(state, action, reward, next_state, done) 
            
            state = next_state 
            total_reward += reward 
 
            # 4. Perform learning step (DQN update)
            if len(replay_buffer) > batch_size: 
                s, a, r, s_next, d = replay_buffer.sample(batch_size) 
 
                # Q-Value prediction for the action taken: Q(s, a)
                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1) 
                
                # TD Target calculation: R + gamma * max Q_target(s', a')
                with torch.no_grad(): 
                    # target_net(s_next).max(1)[0] gets the max Q-value for each next state s'
                    # (1 - d) sets the target to R if the episode is done (d=1)
                    target_q = r + gamma * (1 - d) * target_net(s_next).max(1)[0] 
 
                # Calculate Loss (MSE between predicted Q-value and the TD target)
                loss = nn.MSELoss()(q_values, target_q) 
 
                # Backpropagation
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
 
            if done or truncated: 
                break 
 
        # 5. Update target network periodically (Stabilization)
        if ep % target_update == 0: 
            target_net.load_state_dict(q_net.state_dict()) 
 
        # 6. Epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay) 
        rewards_history.append(total_reward) 
 
        print(f"Episode {ep+1:4d}, Reward: {total_reward:6.1f}, Epsilon: {epsilon:.4f}") 
 
    env.close() 
    return rewards_history 
 
# ------------------------- 
# 4. Run Training and Plot Results 
# ------------------------- 
if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("Starting DQN training on CartPole-v1...")
    
    # [Image of Deep Q-Network Architecture]
    
    rewards = train_dqn() 
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards) 
    plt.xlabel("Episodes") 
    plt.ylabel("Total Reward") 
    plt.title("DQN Training Performance on CartPole-v1") 
    plt.grid(True)
    plt.show()