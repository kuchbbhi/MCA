import numpy as np
import random

# --- 1. Environment Setup and Parameters ---

GRID_SIZE = 4
N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4  # 0: Up, 1: Right, 2: Down, 3: Left
TERMINAL_STATES = [0, 15]

# Learning Parameters
GAMMA = 0.9      # Discount factor
ALPHA = 0.01     # Learning rate (for weight vector w)
N_EPISODES = 100000

# Function Approximation setup
N_FEATURES = N_STATES # Using one-hot encoding, feature dimension equals state dimension

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


# --- 2. Feature Extractor and Value Function ---

def get_features(state):
    """
    Feature extractor: Converts state index into a One-Hot vector (N_FEATURES dimension).
    This is the simplest form of LFA, demonstrating the mechanism.
    """
    feature_vector = np.zeros(N_FEATURES)
    if not is_terminal(state):
        feature_vector[state] = 1.0
    return feature_vector

def estimate_value(features, weights):
    """
    Linear Value Function: V(s, w) = w^T * x(s)
    """
    return np.dot(features, weights)


# --- 3. TD(0) Prediction with Linear Function Approximation ---

def linear_td_prediction(policy, n_episodes):
    """
    Estimates V_pi(s) using TD(0) update rule applied to the weight vector w.
    """
    # Initialize the weight vector (w)
    weights = np.zeros(N_FEATURES)
    
    for episode in range(n_episodes):
        # Start state
        state = random.choice([s for s in range(N_STATES) if not is_terminal(s)])
        
        while not is_terminal(state):
            # Current features and value estimate
            features_t = get_features(state)
            V_t = estimate_value(features_t, weights)
            
            # Interact with environment (following the fixed policy)
            action = policy[state]
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            
            # Next features and value estimate
            features_t_plus_1 = get_features(next_state)
            
            # The next value estimate V(S_t+1) is 0 if the state is terminal
            V_t_plus_1 = 0.0
            if not is_terminal(next_state):
                V_t_plus_1 = estimate_value(features_t_plus_1, weights)
            
            # --- TD Error and Weight Update ---
            
            # TD Target: R + gamma * V(S_t+1, w)
            td_target = reward + GAMMA * V_t_plus_1
            
            # TD Error: TD_Target - V(S_t, w)
            td_error = td_target - V_t
            
            # Weight Update Rule: w <- w + alpha * TD_Error * x(S_t)
            # This is essentially minimizing MSE between V_t and TD_Target
            weights += ALPHA * td_error * features_t
            
            state = next_state
            
        if episode % 10000 == 0:
            print(f"Episode {episode}: Max Value Estimate = {np.max(estimate_value(get_features(s), weights) for s in range(N_STATES) if not is_terminal(s)):.2f}")

    return weights


# --- 4. Execution and Display Results ---

# Define a fixed, high-quality policy to evaluate (e.g., policy found by Q-Learning)
# 0: Up, 1: Right, 2: Down, 3: Left
optimal_policy = np.array([3, 3, 2, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 1, 0])
fixed_policy = optimal_policy

print(f"--- 🚀 TD(0) Prediction with Linear Function Approximation ---")
print(f"Goal: Estimate V_pi(s) for a fixed optimal policy.")
print(f"N_Episodes: {N_EPISODES}, Alpha: {ALPHA}, Gamma: {GAMMA}\n")

final_weights = linear_td_prediction(fixed_policy, N_EPISODES)

# The estimated value V(s) is simply w[s] because of the one-hot encoding feature design.
V_estimated = final_weights

# --- Display Results ---
print("\n--- Final Learned Weight Vector (V(s) estimates) ---")
V_grid = np.round(V_estimated.reshape(GRID_SIZE, GRID_SIZE), 2)
print(V_grid)

print("\nInterpretation:")
print(f"The values represent V(s, w) estimated by the linear model.")
print("The value at state 's' is directly proportional to the weight w[s] because we used a one-hot feature vector.")