import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set device to CPU or CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- 
# 1. Policy Network (Actor)
# ------------------------- 
# Takes state, outputs action probabilities via Softmax
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # Softmax ensures the outputs are valid probabilities that sum to 1
        return torch.softmax(self.net(x), dim=-1)

# ------------------------- 
# 2. Value Network (Baseline / Critic)
# ------------------------- 
# Takes state, outputs the estimated state value V(s)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output a single value: V(s)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------- 
# 3. REINFORCE with Advantage Calculation
# ------------------------- 
def calculate_returns(rewards, gamma):
    """Calculates discounted cumulative returns (G_t) for an episode."""
    returns = []
    G = 0
    # Iterate backwards to calculate returns from the end of the episode
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32).to(DEVICE)

def reinforce_with_advantage(env_name="CartPole-v1", episodes=1000, gamma=0.99):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize Networks
    policy_net = PolicyNetwork(state_dim, action_dim).to(DEVICE)
    value_net = ValueNetwork(state_dim).to(DEVICE)
    
    # Optimizers
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_net.parameters(), lr=1e-3)
    
    rewards_history = []
    
    for ep in range(episodes):
        # 1. Collect Trajectory (Run one full episode)
        states, actions, rewards, log_probs = [], [], [], []
        state, _ = env.reset()
        
        while True:
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            # Sample action
            action_probs = policy_net(state_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Store log probability needed for the gradient update
            log_prob = action_dist.log_prob(action)
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
            
            if done or truncated:
                break

        # Convert collected data to Tensors
        states_t = torch.FloatTensor(np.array(states)).to(DEVICE)
        log_probs_t = torch.stack(log_probs).to(DEVICE)
        returns_t = calculate_returns(rewards, gamma) # G_t

        # 2. Update Value Network (The Baseline V(S_t))
        
        # Predict V(s) for all states in the episode
        predicted_values = value_net(states_t).squeeze(-1)
        
        # Value Loss (MSE between G_t and V(s))
        value_loss = nn.MSELoss()(predicted_values, returns_t)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # 3. Update Policy Network (Actor)
        
        # --- ADVANTAGE FUNCTION CALCULATION ---
        # A_t = G_t - V(S_t)
        # The .detach() is CRITICAL: it prevents gradients from flowing back 
        # into the Value Network during the Policy update.
        advantage = returns_t - predicted_values.detach()
        # 
        
        # Policy Loss (REINFORCE Update using Advantage): - log(pi(A|S)) * Advantage
        # This is the Policy Gradient theorem's sample approximation
        policy_loss = -(log_probs_t * advantage).mean()
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # 4. Logging and Tracking
        total_reward = sum(rewards)
        rewards_history.append(total_reward)
        
        if ep % 50 == 0:
            print(f"Episode {ep:4d}, Reward: {total_reward:6.1f}, Advantage (Policy) Loss: {policy_loss.item():.4f}")
            
        # Stopping condition for CartPole
        if len(rewards_history) > 100 and np.mean(rewards_history[-100:]) >= 475:
            print(f"\nEnvironment solved in {ep} episodes!")
            break

    env.close()
    return rewards_history

# ------------------------- 
# 4. Run Training and Plot Results 
# ------------------------- 
if __name__ == '__main__':
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    # The default random seed is handled by Python's random module, 
    # but since PyTorch/NumPy are set, it's mostly covered.
    
    print("Starting REINFORCE with Advantage Function training on CartPole-v1...")
    
    rewards = reinforce_with_advantage() 
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards) 
    plt.xlabel("Episodes") 
    plt.ylabel("Total Reward") 
    plt.title("REINFORCE with Advantage Function Training Performance") 
    plt.grid(True)
    plt.show()