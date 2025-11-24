import numpy as np

# --- 1. Define the Environment and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = [0, 15]
GAMMA = 0.9      # Discount factor
ALPHA = 0.1      # Learning rate
N_EPISODES = 10000

# Rewards: -1 for every non-terminal step
STEP_REWARD = -1

# All possible actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [0, 1, 2, 3]


def is_terminal(state):
    """Checks if a state is terminal."""
    return state in TERMINAL_STATES

def get_next_state(state, action):
    """Calculates the next state given current state and action."""
    row, col = state // GRID_SIZE, state % GRID_SIZE

    if action == 0:  # Up
        new_row, new_col = max(0, row - 1), col
    elif action == 1:  # Right
        new_row, new_col = row, min(GRID_SIZE - 1, col + 1)
    elif action == 2:  # Down
        new_row, new_col = min(GRID_SIZE - 1, row + 1), col
    elif action == 3:  # Left
        new_row, new_col = row, max(0, col - 1)
    
    return new_row * GRID_SIZE + new_col


# --- 2. TD(0) Policy Evaluation Algorithm ---

def td_zero_policy_evaluation():
    # Initialize V(s) to 0 for all non-terminal states
    V = np.zeros(N_STATES)
    
    for _ in range(N_EPISODES):
        # Start state is chosen randomly from non-terminal states
        state = np.random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        
        while not is_terminal(state):
            # 1. Agent follows the fixed equiprobable random policy
            action = np.random.choice(ACTIONS)
            
            # 2. Interact with environment
            next_state = get_next_state(state, action)
            
            # 3. Observe Reward (Reward is for the step taken)
            reward = STEP_REWARD
            
            # --- TD(0) Update Rule ---
            
            # The value of a terminal state is its immediate reward (0 in this setup since reward is -1 per step)
            # However, since we track the reward *per step*, we use 0 for terminal V(s')
            V_next = 0 if is_terminal(next_state) else V[next_state]
            
            # Calculate the TD Error: [Reward + Gamma * V(s') - V(s)]
            td_target = reward + GAMMA * V_next
            td_error = td_target - V[state]
            
            # Update V(s)
            V[state] += ALPHA * td_error
            
            # Move to the next state
            state = next_state
            
    return V


# --- 3. Execute and Display Results ---
estimated_V = td_zero_policy_evaluation()

print("🏁 Estimated Value Function V(s) using TD(0) Policy Evaluation:\n")

# Format the V(s) results into a 4x4 grid for easy viewing
V_grid = np.round(estimated_V.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)