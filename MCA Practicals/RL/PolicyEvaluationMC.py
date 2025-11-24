import numpy as np

# --- 1. Define the Grid World and Parameters ---

# 4x4 Grid: States 0 and 15 are terminal.
GRID_SIZE = 4
TERMINAL_STATES = [0, 15]
GAMMA = 0.9
N_EPISODES = 10000

# Rewards: -1 for every non-terminal step
REWARDS = -1

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
    
    # New state index
    return new_row * GRID_SIZE + new_col


# --- 2. Monte Carlo First-Visit Policy Evaluation ---

def mc_first_visit_policy_evaluation():
    # Initialize V(s) to 0 for all non-terminal states
    V = np.zeros(GRID_SIZE * GRID_SIZE)
    # Stores all returns for a state to calculate the average
    Returns = {s: [] for s in range(GRID_SIZE * GRID_SIZE)}
    
    for _ in range(N_EPISODES):
        # Generate an Episode: (State, Action, Reward, Next_State)
        episode = []
        # Start state is chosen randomly from non-terminal states
        state = np.random.choice([s for s in range(GRID_SIZE * GRID_SIZE) if not is_terminal(s)])

        while not is_terminal(state):
            # Fixed equiprobable random policy (0.25 for each action)
            action = np.random.choice(ACTIONS)
            next_state = get_next_state(state, action)
            reward = REWARDS
            
            episode.append((state, action, reward))
            state = next_state
        
        # Policy Evaluation part (Backward Pass)
        G = 0 # Return accumulator
        visited_states = set()
        
        # Iterate backwards through the episode
        for t in reversed(range(len(episode))):
            state_t, _, reward_t = episode[t]
            
            # Calculate the return G_t
            G = reward_t + GAMMA * G
            
            # Monte Carlo First-Visit: Update V(s) only if state_t is visited for the FIRST time in this episode
            if state_t not in visited_states:
                # Add the return to the list for this state
                Returns[state_t].append(G)
                
                # New estimate is the average of all collected returns
                V[state_t] = np.mean(Returns[state_t])
                
                # Mark as visited
                visited_states.add(state_t)
                
    return V


# --- 3. Execute and Display Results ---
estimated_V = mc_first_visit_policy_evaluation()

print("Estimated Value Function V(s) after 10,000 episodes:\n")

# Format the V(s) results into a 4x4 grid for easy viewing
V_grid = np.round(estimated_V.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)