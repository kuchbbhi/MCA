import numpy as np

# --- 1. Define the Environment and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = [0, 15]
GAMMA = 0.9      # Discount factor
ALPHA = 0.1      # Learning rate
LAMBDA = 0.8     # The TD(lambda) parameter (0 for TD(0), 1 for MC)
N_EPISODES = 10000

# Rewards: -1 for every non-terminal step
STEP_REWARD = -1
ACTIONS = [0, 1, 2, 3] # Fixed equiprobable policy

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


# --- 2. TD(lambda) Policy Evaluation Algorithm ---

def td_lambda_policy_evaluation():
    # Initialize V(s) to 0
    V = np.zeros(N_STATES)
    
    for _ in range(N_EPISODES):
        # Initialize Eligibility Traces E(s) at the beginning of each episode
        E = np.zeros(N_STATES)
        
        # Start state is chosen randomly from non-terminal states
        state = np.random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        
        while not is_terminal(state):
            # 1. Agent follows the fixed equiprobable random policy
            action = np.random.choice(ACTIONS)
            
            # 2. Interact with environment
            next_state = get_next_state(state, action)
            
            # 3. Observe Reward (Reward is for the step taken)
            reward = STEP_REWARD
            
            # --- TD(lambda) Update Step ---
            
            # Get the value of the next state (0 if terminal)
            V_next = 0 if is_terminal(next_state) else V[next_state]
            
            # 1. Calculate the TD Error (Delta)
            td_error = reward + GAMMA * V_next - V[state]
            
            # 2. Update the Eligibility Trace for the current state (Accumulating Trace)
            E[state] += 1
            
            # 3. Update V(s) for ALL states based on their trace
            V += ALPHA * td_error * E
            
            # 4. Decay the Eligibility Trace for all states
            E *= GAMMA * LAMBDA
            
            # Move to the next state
            state = next_state
            
    return V


# --- 3. Execute and Display Results ---
estimated_V = td_lambda_policy_evaluation()

print(f"🏁 Estimated Value Function V(s) using TD({LAMBDA}) Policy Evaluation:\n")

# Format the V(s) results into a 4x4 grid for easy viewing
V_grid = np.round(estimated_V.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)