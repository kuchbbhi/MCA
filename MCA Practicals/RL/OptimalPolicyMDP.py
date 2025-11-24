import numpy as np

# --- 1. Define the Environment and Optimal Policy ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
TERMINAL_STATES = [0, 15]
GAMMA = 0.9      # Discount factor
ACTIONS = [0, 1, 2, 3]  # 0: Up, 1: Right, 2: Down, 3: Left

# Optimal Policy from previous Value Iteration (State 15 = +10, State 0 = -10)
# This is hard-coded based on the expected solution for the asymmetric MDP.
# 0 1 2 3
# 4 5 6 7
# 8 9 10 11
# 12 13 14 15
OPTIMAL_POLICY = np.array([
    -1,  2,  2,  2,  # -1 is T (State 0)
     1,  1,  1,  2,
     1,  1,  1,  2,
     1,  1,  1, -1   # -1 is T (State 15)
], dtype=int)


def get_reward(state):
    """Calculates the reward for reaching a state (Terminal states give final reward)."""
    if state == 15:
        return 10.0  # Goal state
    elif state == 0:
        return -10.0 # Bad end state
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
    # ... (omitted boundary checks for brevity, same as previous code)
    elif action == 1:  # Right
        new_row, new_col = row, min(GRID_SIZE - 1, col + 1)
    elif action == 2:  # Down
        new_row, new_col = min(GRID_SIZE - 1, row + 1), col
    elif action == 3:  # Left
        new_row, new_col = row, max(0, col - 1)
    
    return new_row * GRID_SIZE + new_col


# --- 2. Rollout Simulation ---

def rollout_optimal_policy(start_state):
    state = start_state
    trajectory = [state]
    total_return = 0
    discount_factor = 1.0 # Initial $\gamma^0 = 1$
    
    # Check if the start state is terminal
    if is_terminal(state):
        print(f"Start state {state} is terminal. Return is {get_reward(state):.2f}")
        return trajectory, get_reward(state)

    while not is_terminal(state):
        # 1. Select the action based on the optimal policy
        action = OPTIMAL_POLICY[state]
        
        # 2. Transition to the next state
        next_state = get_next_state(state, action)
        
        # 3. Determine the reward for the transition
        if is_terminal(next_state):
            # Reward is the final reward for reaching the terminal state
            reward = get_reward(next_state) 
        else:
            # Reward is the step cost
            reward = -1.0 
        
        # 4. Update the total return (G)
        total_return += discount_factor * reward
        
        # 5. Update state and discount factor
        state = next_state
        discount_factor *= GAMMA
        
        trajectory.append(state)

    return trajectory, total_return

# --- 3. Execute and Display Results ---

# Starting the simulation from State 1 (position (0,1))
# Since State 0 (0,0) is a terminal state in the MDP, we start next to it to show a path.
START_STATE = 1 

trajectory, total_return = rollout_optimal_policy(START_STATE)

print(f"✅ Rollout starting from state {START_STATE} (0,1) following $\pi^*$ (Goal: 15):\n")
print(f"Path: {trajectory}")
print(f"Total Discounted Return (G): {total_return:.4f}")

print("\n---")
print("Policy Used (Optimal Action from State):")
actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
path_actions = []
# Print the action taken from each non-terminal state in the path
for i in range(len(trajectory) - 1):
    state = trajectory[i]
    action_taken = OPTIMAL_POLICY[state]
    path_actions.append(f"{state} ({actions_map.get(action_taken, '?')})")

print(f"State (Action): {' -> '.join(path_actions)}")