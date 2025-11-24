import numpy as np

# --- 1. Define the Environment and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4
TERMINAL_STATES = [0, 15]
GAMMA = 0.9      # Discount factor
ALPHA = 0.1      # Learning rate
EPSILON = 0.9    # Initial exploration rate
EPSILON_DECAY = 0.9999 # Decay rate per step
MIN_EPSILON = 0.01 # Minimum exploration rate
N_EPISODES = 20000 

ACTIONS = [0, 1, 2, 3]  # 0: Up, 1: Right, 2: Down, 3: Left


def get_reward(state):
    """Calculates the reward received for a transition leading to the state."""
    if state == 15:
        return 10.0  # Goal state
    elif state == 0:
        return -10.0 # Bad state
    else:
        return -1.0  # Step cost

def is_terminal(state):
    """Checks if a state is terminal."""
    return state in TERMINAL_STATES

def get_next_state(state, action):
    """Calculates the next state given current state and action (Deterministic transition)."""
    row, col = state // GRID_SIZE, state % GRID_SIZE
    # (Boundary checks omitted for brevity, same logic as before)
    if action == 0:  # Up
        new_row, new_col = max(0, row - 1), col
    elif action == 1:  # Right
        new_row, new_col = row, min(GRID_SIZE - 1, col + 1)
    elif action == 2:  # Down
        new_row, new_col = min(GRID_SIZE - 1, row + 1), col
    elif action == 3:  # Left
        new_row, new_col = row, max(0, col - 1)
    
    return new_row * GRID_SIZE + new_col

# --- 2. Exploration Policy (Epsilon-Greedy) ---

def choose_action(state, Q_table, epsilon):
    """Chooses action a using epsilon-greedy policy."""
    if np.random.random() < epsilon:
        # Explore: choose random action
        return np.random.choice(N_ACTIONS)
    else:
        # Exploit: choose action with max Q-value
        # Use Q_table[state, :] to get all Q-values for the current state
        return np.argmax(Q_table[state, :])


# --- 3. Q-Learning Algorithm ---

def q_learning():
    # Initialize Q-table to zeros
    Q = np.zeros((N_STATES, N_ACTIONS))
    current_epsilon = EPSILON

    for episode in range(N_EPISODES):
        # Start state is chosen randomly from non-terminal states
        state = np.random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        
        while not is_terminal(state):
            # 1. Choose action using epsilon-greedy
            action = choose_action(state, Q, current_epsilon)
            
            # 2. Interact with environment
            next_state = get_next_state(state, action)
            
            # 3. Observe Reward (Reward is for reaching next_state)
            reward = get_reward(next_state) if is_terminal(next_state) else -1.0
            
            # --- Q-Learning Update Rule ---
            
            # Q-Learning is off-policy: it uses the max Q-value of the next state
            # Find the best action a' in the next state s'
            max_next_q = np.max(Q[next_state, :])
            
            # Calculate the TD target
            td_target = reward + GAMMA * max_next_q
            
            # Update the Q-value for (s, a)
            Q[state, action] += ALPHA * (td_target - Q[state, action])
            
            # Move to the next state
            state = next_state

        # Epsilon decay to reduce exploration over time
        current_epsilon = max(MIN_EPSILON, current_epsilon * EPSILON_DECAY)
            
    return Q


# --- 4. Extract Optimal Policy $\pi^*(s)$ and Display Results ---

def extract_optimal_policy(Q):
    policy = np.zeros(N_STATES, dtype=int)
    for state in range(N_STATES):
        if is_terminal(state):
            continue
        # The optimal action is the one with the maximum Q-value
        policy[state] = np.argmax(Q[state, :])
    return policy


# --- 5. Execution ---
optimal_Q = q_learning()
optimal_policy = extract_optimal_policy(optimal_Q)

print("🏁 Optimal Q-Table Learned by Q-Learning:\n")
# Display the max Q-value for each state, which estimates V*(s)
V_star_estimate = np.max(optimal_Q, axis=1)
V_grid = np.round(V_star_estimate.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)

print("\n\n🏆 Optimal Policy $\pi^*(s)$ (0: Up, 1: Right, 2: Down, 3: Left):\n")
policy_map = {0: '↑', 1: '→', 2: '↓', 3: '←'} 
policy_grid = []
for i, a in enumerate(optimal_policy):
    if i == 0: policy_grid.append('S')
    elif i == 15: policy_grid.append('G')
    else: policy_grid.append(policy_map[a])

policy_grid = np.array(policy_grid).reshape(GRID_SIZE, GRID_SIZE)
print(policy_grid)