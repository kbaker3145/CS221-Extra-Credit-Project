"""
File Name: Random Action Algorithm

This algorithm randomly samples actions. 

"""

import numpy as np
import random
import matplotlib.pyplot as plt
import csv

# Constants and Hyperparameters
NUM_MONTHS = 12 # or 60 for 5 years

# Financial parameters
SAVINGS_GROWTH_RATE = 0.005  # 0.5% growth per month
DEBT_GROWTH_RATE = 0.01  # 1% growth per month
PROMOTION_PROB = 0.10  # 10% chance for promotion
LAYOFF_PROB = 0.05  # 5% chance of layoff
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

# Simulation with random actions
def simulate_random_actions(num_months):
    current_state = initial_state.copy()  # Reset the state for each episode
    total_reward = 0

    for month in range(num_months):
        # Select a random action
        action = random.choice(actions)
        
        # Transition to the next state based on the action taken
        current_state = transition(current_state, action)
        
        # Calculate reward based on the new state
        monthly_reward = reward(current_state)
        
        # Accumulate total reward
        total_reward += monthly_reward

    return total_reward, current_state

# Evaluation of random actions over multiple episodes
NUM_EPISODES = 10
cumulative_rewards = []
debt_over_time = []
savings_over_time = []

for episode in range(NUM_EPISODES):
    # Run a single episode with random actions and record results
    total_reward, final_state = simulate_random_actions(NUM_MONTHS)
    
    # Track cumulative reward, debt, and savings for analysis
    cumulative_rewards.append(total_reward)
    debt_over_time.append(final_state["debt"])
    savings_over_time.append(final_state["savings"])

# Plot cumulative rewards and save as image
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode (Random Actions - Baseline)")
plt.savefig("random_cumulative_rewards.png")

# Plot debt and savings over episodes and save as image
plt.figure(figsize=(10, 6))
plt.plot(debt_over_time, label="Debt")
plt.plot(savings_over_time, label="Savings")
plt.xlabel("Episode")
plt.ylabel("Balance")
plt.title("Debt and Savings Over Episodes (Random Actions - Baseline)")
plt.legend()
plt.savefig("random_debt_savings_over_time.png")

# Save random cumulative rewards to CSV file for comparison
with open("random_cumulative_rewards.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Cumulative Reward"])  # Header
    for episode, reward in enumerate(cumulative_rewards):
        writer.writerow([episode, reward])

# Print final statistics for performance evaluation
print(f"Average Reward per Episode: {np.mean(cumulative_rewards)}")
print(f"Final Debt after {NUM_EPISODES} Episodes: {debt_over_time[-1]}")
print(f"Final Savings after {NUM_EPISODES} Episodes: {savings_over_time[-1]}")
