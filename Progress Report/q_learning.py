"""
File Name: Q-Learning

This algorithm learns the best policy.

"""
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import csv

# Constants and Hyperparameters
NUM_MONTHS = 12  # or 60 for 5 years
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON_INITIAL = 0.3  # Initial epsilon for exploration
EPSILON_DECAY = 0.99   # Decay rate for epsilon per episode

# Financial parameters
SAVINGS_GROWTH_RATE = 0.005  # 0.5% growth per month
DEBT_GROWTH_RATE = 0.01  # 1% growth per month
PROMOTION_PROB = 0.10  # 10% chance for promotion
LAYOFF_PROB = 0.05  # 5% chance for layoff
MEDICAL_EXPENSE_PROB = 0.02  # 2% chance of medical emergency
HOME_REPAIR_PROB = 0.03  # 3% chance of home repair
MEDICAL_COST = 5000
HOME_REPAIR_COST = 2000
PROMOTION_INCREASE = 1.15
LAYOFF_DECREASE = 0.5

# Initial conditions
initial_state = {
    "savings": 10000,  # example value
    "debt": 5000,      # example value
    "income": 4000,    # example monthly income
    "expenses": 2000   # example monthly expenses
}

actions = ["allocate_savings", "pay_debt", "increase_spending"]
Q = {}  # Q-table

# Transition function
def transition(state, action):
    # Copy the state dictionary to avoid modifying the original
    state = state.copy()
    
    # Apply financial growth
    state["savings"] += state["savings"] * SAVINGS_GROWTH_RATE
    state["debt"] += state["debt"] * DEBT_GROWTH_RATE

    # Handle income fluctuations
    income_event = random.random()
    if income_event < PROMOTION_PROB:
        state["income"] *= PROMOTION_INCREASE
    elif income_event < (PROMOTION_PROB + LAYOFF_PROB):
        state["income"] *= LAYOFF_DECREASE

    # Handle unexpected expenses
    if random.random() < MEDICAL_EXPENSE_PROB:
        state["savings"] -= MEDICAL_COST
    if random.random() < HOME_REPAIR_PROB:
        state["savings"] -= HOME_REPAIR_COST

    # Action effects
    if action == "allocate_savings":
        # Allocate remaining income to savings after expenses
        state["savings"] += state["income"] - state["expenses"]
    elif action == "pay_debt":
        # Pay debt with remaining income after expenses
        payment = min(state["debt"], state["income"] - state["expenses"])
        state["debt"] -= payment
        state["savings"] += state["income"] - state["expenses"] - payment
    elif action == "increase_spending":
        # Increase expenses by 5%
        state["expenses"] *= 1.05

    return state

# Reward function to encourage low debt and high savings
def reward(state):
    debt_penalty = -0.2 * state["debt"]
    savings_reward = 0.1 * state["savings"]
    total_reward = savings_reward + debt_penalty
    return total_reward

# Choose action using epsilon-greedy policy
def choose_action(state, epsilon):
    # Convert the state to a hashable tuple for Q-table lookup
    state_tuple = tuple(state.values())
    
    # Explore or exploit
    if random.random() < epsilon or state_tuple not in Q:
        return random.choice(actions)
    else:
        return max(Q[state_tuple], key=Q[state_tuple].get)

# Q-learning update function
def update_q_table(prev_state, action, reward, next_state):
    # Convert states to tuples for Q-table storage
    prev_state_tuple = tuple(prev_state.values())
    next_state_tuple = tuple(next_state.values())
    
    # Initialize Q-values for new states if they don’t exist in the Q-table
    if prev_state_tuple not in Q:
        Q[prev_state_tuple] = {a: 0.0 for a in actions}
    if next_state_tuple not in Q:
        Q[next_state_tuple] = {a: 0.0 for a in actions}

    # Q-learning update: Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + gamma * max(Q(s', a')))
    best_next_action = max(Q[next_state_tuple].values())  # max_a' Q(s', a')
    
    # Update the Q-value
    Q[prev_state_tuple][action] = ((1 - LEARNING_RATE) * Q[prev_state_tuple][action] + 
                                   LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next_action))

# Q-learning simulation for each episode
def simulate_q_learning(num_months, epsilon):
    current_state = initial_state.copy()  # Reset the state for each episode
    total_reward = 0

    for month in range(num_months):
        # Select an action using epsilon-greedy policy
        action = choose_action(current_state, epsilon)
        
        # Copy current state for Q-learning update reference
        prev_state = current_state.copy()
        
        # Transition to the next state based on the action taken
        current_state = transition(current_state, action)
        
        # Calculate reward based on the new state
        monthly_reward = reward(current_state)
        
        # Update Q-table based on observed transition
        update_q_table(prev_state, action, monthly_reward, current_state)
        
        # Accumulate total reward
        total_reward += monthly_reward

    return total_reward, current_state

# Evaluation of Q-learning over multiple episodes
NUM_EPISODES = 10
cumulative_rewards = []
debt_over_time = []
savings_over_time = []
q_value_changes = []
prev_Q = {}

for episode in range(NUM_EPISODES):
    # Decay epsilon over episodes to reduce exploration gradually
    epsilon = max(0.1, EPSILON_INITIAL * (EPSILON_DECAY ** episode))
    
    # Run a single episode and record results
    total_reward, final_state = simulate_q_learning(NUM_MONTHS, epsilon)
    
    # Track cumulative reward, debt, and savings for analysis
    cumulative_rewards.append(total_reward)
    debt_over_time.append(final_state["debt"])
    savings_over_time.append(final_state["savings"])

    # Track Q-value changes for convergence analysis
    q_change = 0
    for state_tuple in Q:
        for action in Q[state_tuple]:
            if state_tuple in prev_Q and action in prev_Q[state_tuple]:
                q_change += abs(Q[state_tuple][action] - prev_Q[state_tuple][action])
    q_value_changes.append(q_change)

    # Make a deep copy of Q for prev_Q to ensure independence
    prev_Q = copy.deepcopy(Q)

# Plot cumulative rewards and save as image
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode (Reward Analysis)")
plt.savefig("cumulative_rewards.png")

# Plot debt and savings over episodes and save as image
plt.figure(figsize=(10, 6))
plt.plot(debt_over_time, label="Debt")
plt.plot(savings_over_time, label="Savings")
plt.xlabel("Episode")
plt.ylabel("Balance")
plt.title("Debt and Savings Over Episodes (Financial Stability)")
plt.legend()
plt.savefig("q_learning_debt_savings_over_time.png")

# Plot Q-value changes to observe policy convergence and save as image
plt.figure(figsize=(10, 6))
plt.plot(q_value_changes)
plt.xlabel("Episode")
plt.ylabel("Q-value Change")
plt.title("Q-value Changes Over Episodes (Policy Convergence)")
plt.savefig("q_learning_q_value_changes.png")

# Save Q-values to CSV file
with open("q_learning_q_values.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["State", "Action", "Q-Value"])  # Header
    for state_tuple, actions_dict in Q.items():
        for action, q_value in actions_dict.items():
            writer.writerow([state_tuple, action, q_value])

# Print final statistics for performance evaluation
print(f"Average Reward per Episode: {np.mean(cumulative_rewards)}")
print(f"Final Debt after {NUM_EPISODES} Episodes: {debt_over_time[-1]}")
print(f"Final Savings after {NUM_EPISODES} Episodes: {savings_over_time[-1]}")
print(f"Total Q-value Change in Final Episode: {q_value_changes[-1]}")
