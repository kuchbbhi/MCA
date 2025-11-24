import numpy as np
import random

# --- 1. Environment Setup and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4  # 0: Up, 1: Right, 2: Down, 3: Left
TERMINAL_STATES = [0, 15]

GAMMA = 0.9      # Discount factor
EPISODES_PREDICTION = 50000 
EPISODES_CONTROL = 100000

# Control Parameters
EPSILON = 0.1    # Epsilon-soft policy for control

def get_reward(state):
    """Reward function based on the state reached."""
    if state == 15:
        return 10.0  # Goal state (positive reward)
    elif state == 0:
        return -10.0 # Bad state (negative reward)
    else:
        return -1.0  # Step cost

def is_terminal(state):
    """Checks if a state is terminal."""
    return state in TERMINAL_STATES

def get_next_state(state, action):
    """Calculates the next state given current state and action (Deterministic transition)."""
    row, col = state // GRID_SIZE, state % GRID_SIZE
    
    # Action mapping: 0: Up, 1: Right, 2: Down, 3: Left
    if action == 0: new_row, new_col = max(0, row - 1), col
    elif action == 1: new_row, new_col = row, min(GRID_SIZE - 1, col + 1)
    elif action == 2: new_row, new_col = min(GRID_SIZE - 1, row + 1), col
    elif action == 3: new_row, new_col = row, max(0, col - 1)
    
    return new_row * GRID_SIZE + new_col


# --- 2. Monte Carlo Prediction (Policy Evaluation) ---

def mc_prediction(policy, n_episodes):
    """Estimates the state-value function V(s) for a given policy using First-Visit MC."""
    
    V = np.zeros(N_STATES)
    returns = {s: [] for s in range(N_STATES)}  # Store returns for averaging
    
    for _ in range(n_episodes):
        # Generate an episode (a sequence of S, A, R)
        episode = []
        state = random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        
        while not is_terminal(state):
            action = policy[state]
            next_state = get_next_state(state, action)
            reward = get_reward(next_state) if is_terminal(next_state) else -1.0
            episode.append((state, action, reward))
            state = next_state
        
        # Calculate returns (G_t) and update V(s) using First-Visit MC
        G = 0
        visited_states = set()
        
        # Iterate backwards through the episode to calculate G_t
        for t in reversed(range(len(episode))):
            state_t, _, reward_t = episode[t]
            
            # G_t is the discounted return from time t
            G = reward_t + GAMMA * G
            
            # First-Visit MC: Only update V(s) if this is the first time state_t was visited
            if state_t not in visited_states:
                returns[state_t].append(G)
                V[state_t] = np.mean(returns[state_t])
                visited_states.add(state_t)
                
    return V


# --- 3. Monte Carlo Control (On-Policy) ---

def choose_action_epsilon_soft(state, Q_table, epsilon):
    """Chooses action a using the epsilon-soft policy."""
    if np.random.random() < epsilon:
        # Explore: choose random action
        return np.random.choice(N_ACTIONS)
    else:
        # Exploit: choose action with max Q-value (Greedy action)
        return np.argmax(Q_table[state, :])

def mc_control_on_policy(n_episodes, epsilon):
    """On-Policy MC Control (Every-Visit) to estimate Q* and pi*."""
    
    # Initialize Q(s, a) arbitrarily (e.g., to zeros)
    Q = np.zeros((N_STATES, N_ACTIONS))
    
    # Store returns for averaging
    returns = {(s, a): [] for s in range(N_STATES) for a in range(N_ACTIONS)}
    
    for _ in range(n_episodes):
        # Start state
        state = random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        episode = []
        
        # 1. Generate an episode using the current epsilon-soft policy
        while not is_terminal(state):
            action = choose_action_epsilon_soft(state, Q, epsilon)
            next_state = get_next_state(state, action)
            reward = get_reward(next_state) if is_terminal(next_state) else -1.0
            episode.append((state, action, reward))
            state = next_state

        # 2. Policy Evaluation (Update Q(s, a))
        G = 0
        
        # Iterate backwards through the episode
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = reward_t + GAMMA * G
            
            # Every-Visit MC: Update Q(s, a) every time the pair is encountered
            returns[(state_t, action_t)].append(G)
            Q[state_t, action_t] = np.mean(returns[(state_t, action_t)])
            
        # 3. Policy Improvement is implicit: The policy used in the next episode 
        # is the epsilon-greedy version of the updated Q (choose_action_epsilon_soft).
                
    return Q


# --- 4. Execution and Display Results ---

# --- A. MC Prediction (Requires a fixed policy) ---
# Define a fixed policy (e.g., always go Right, action=1)
fixed_policy = np.ones(N_STATES, dtype=int) * 1 

print(f"--- 🚀 MC Prediction (V(s)) for a fixed policy (Always Go Right) ---")
V_predicted = mc_prediction(fixed_policy, EPISODES_PREDICTION)
V_grid = np.round(V_predicted.reshape(GRID_SIZE, GRID_SIZE), 2)
print(f"V(s) estimated after {EPISODES_PREDICTION} episodes (Gamma={GAMMA}):\n")
print(V_grid)


# --- B. MC Control (Finds the optimal policy pi*) ---

print(f"\n--- 🏆 MC Control (Q*, pi*) using On-Policy Epsilon-Soft ---")
Q_optimal = mc_control_on_policy(EPISODES_CONTROL, EPSILON)

# Extract the optimal policy (Greedy policy w.r.t the final Q-table)
optimal_policy = np.argmax(Q_optimal, axis=1)

print(f"Optimal Q-Table (Max Q-value V*(s)) estimated after {EPISODES_CONTROL} episodes:\n")
V_star_estimate = np.max(Q_optimal, axis=1)
V_grid_opt = np.round(V_star_estimate.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid_opt)

print("\nOptimal Policy $\pi^*(s)$ (0: ↑, 1: →, 2: ↓, 3: ←):\n")
policy_map = {0: '↑', 1: '→', 2: '↓', 3: '←'} 
policy_grid = []
for i, a in enumerate(optimal_policy):
    if i == 0: policy_grid.append('X') # Start 
    elif i == 15: policy_grid.append('G') # Goal
    else: policy_grid.append(policy_map[a])

policy_grid_final = np.array(policy_grid).reshape(GRID_SIZE, GRID_SIZE)
print(policy_grid_final)