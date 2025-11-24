import numpy as np

# --- 1. Define the Modified MDP and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = [0, 15]
GAMMA = 0.9      # Discount factor
THETA = 1e-4     # Convergence threshold
ACTIONS = [0, 1, 2, 3]  # 0: Up, 1: Right, 2: Down, 3: Left


def get_reward(state):
    """Calculates the reward for reaching a state."""
    if state == 15:
        return 10.0  # Goal state (positive reward)
    elif state == 0:
        return -10.0 # Bad state / Start point (negative reward)
    else:
        return -1.0  # Step cost

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


# --- 2. Value Iteration Algorithm (Unchanged Logic) ---

def value_iteration():
    V = np.zeros(N_STATES)
    
    while True:
        delta = 0
        V_new = V.copy()
        
        for state in range(N_STATES):
            if is_terminal(state):
                # Terminal states have a fixed value equal to their immediate reward
                # V[state] is set to the reward for reaching it.
                V_new[state] = get_reward(state)
                continue
            
            v_old = V[state]
            
            q_values = []
            for action in ACTIONS:
                next_state = get_next_state(state, action)
                
                # Reward is for *reaching* the next state
                reward = get_reward(next_state) if is_terminal(next_state) else -1.0
                
                q_value = reward + GAMMA * V[next_state]
                q_values.append(q_value)
            
            V_new[state] = np.max(q_values)
            delta = max(delta, np.abs(v_old - V_new[state]))
        
        V = V_new
        
        if delta < THETA:
            break
            
    return V


# --- 3. Extract Optimal Policy $\pi^*(s)$ ---

def extract_optimal_policy(V):
    policy = np.zeros(N_STATES, dtype=int)
    
    for state in range(N_STATES):
        if is_terminal(state):
            continue
            
        q_values = []
        for action in ACTIONS:
            next_state = get_next_state(state, action)
            
            # Use the same reward logic as in Value Iteration
            reward = get_reward(next_state) if is_terminal(next_state) else -1.0
            
            q_value = reward + GAMMA * V[next_state]
            q_values.append(q_value)
        
        best_action = np.argmax(q_values)
        policy[state] = best_action
        
    return policy

# --- 4. Execute and Display Results ---
optimal_V = value_iteration()
optimal_policy = extract_optimal_policy(optimal_V)

print("Optimal Value Function V*(s) (State 15 = +10, State 0 = -10):\n")
V_grid = np.round(optimal_V.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)

print("\n\nOptimal Policy $\pi^*(s)$ (0: Up, 1: Right, 2: Down, 3: Left):\n")
policy_map = {0: '↑', 1: '→', 2: '↓', 3: '←'} 
# Use 'S' for State 0 (Start/Bad End) and 'G' for State 15 (Goal)
policy_grid = []
for i, a in enumerate(optimal_policy):
    if i == 0: policy_grid.append('S')
    elif i == 15: policy_grid.append('G')
    else: policy_grid.append(policy_map[a])

policy_grid = np.array(policy_grid).reshape(GRID_SIZE, GRID_SIZE)
print(policy_grid)