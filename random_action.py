"""
File Name: Random Action Algorithm

This algorithm randomly samples actions with a happiness factor.
"""

import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import csv

# Constants and Hyperparameters
NUM_MONTHS = 60  

# Financial parameters
SAVINGS_GROWTH_RATE = 0.005  # 0.5% growth per month
DEBT_GROWTH_RATE = 0.01  # 1% growth per month
PROMOTION_PROB = 0.10  # 10% chance for promotion
LAYOFF_PROB = 0.05  # 5% chance of layoff
MEDICAL_EXPENSE_PROB = 0.02  # 2% chance of medical emergency
HOME_REPAIR_PROB = 0.03  # 3% chance of home repair
MEDICAL_COST = 1000
HOME_REPAIR_COST = 200
PROMOTION_INCREASE = 1.15
LAYOFF_DECREASE = 0.8

# Parse arguments from the terminal for initial state values
parser = argparse.ArgumentParser(description="Random Action Baseline for Budget Optimization")
parser.add_argument("--income", type=float, default=4000, help="Initial monthly income")
parser.add_argument("--debt", type=float, default=5000, help="Initial debt")
parser.add_argument("--savings", type=float, default=10000, help="Initial savings")
parser.add_argument("--expenses", type=float, default=2000, help="Initial monthly expenses")
args = parser.parse_args()

# Initial conditions
initial_state = {
    "savings": args.savings,
    "debt": args.debt,
    "income": args.income,
    "expenses": args.expenses
}

actions = ["allocate_savings", "pay_debt", "increase_spending"]

# Transition function
def transition(state, action):
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

   # if there is no disposable income it should become debt regardless of the action taken
    state["disposable_income"] = state["income"] - state["expenses"]
    if state["disposable_income"] < 0:
         state["debt"] -= state["disposable_income"]
    else:
        if action == "allocate_savings":
            state["savings"] += state["disposable_income"]
        elif action == "pay_debt":
            # if paying off debt is chosen, the debt should be paid off as much as it can and then the rest should be saved
            payment = min(abs(state["debt"]), state["disposable_income"])
            state["debt"] -= payment 
            state["savings"] += (state["disposable_income"] - payment)
            # if increase_spending is chosen then expenses are multiplied by 5%
        elif action == "increase_spending":  
            state["expenses"] *= 1.05

    return state

# Reward function with happiness factor
def reward(state):
    savings_reward = 5000 * state["savings"]
    debt_penalty = -1000 * state["debt"]
    
    # Happiness factor: Reward for spending within a reasonable range
    happiness_reward = 0.05 * state["expenses"]
    
    total_reward = savings_reward + debt_penalty + happiness_reward
    return total_reward, happiness_reward

# Simulation with random actions
def simulate_random_actions(num_months):
    current_state = initial_state.copy()
    total_reward = 0
    total_happiness = 0  # Track cumulative happiness factor

    for month in range(num_months):
        # Select a random action
        action = random.choice(actions)
        
        # Transition to the next state
        current_state = transition(current_state, action)
        
        # Calculate reward and happiness factor
        monthly_reward, happiness_reward = reward(current_state)
        
        # Accumulate totals
        total_reward += monthly_reward
        total_happiness += happiness_reward

    return total_reward, total_happiness, current_state

# Evaluation of random actions over multiple episodes
NUM_EPISODES = 1
cumulative_rewards = []
happiness_factors = []
debt_over_time = []
savings_over_time = []

for episode in range(NUM_EPISODES):
    total_reward, total_happiness, final_state = simulate_random_actions(NUM_MONTHS)
    
    # Track results
    cumulative_rewards.append(total_reward)
    happiness_factors.append(total_happiness)
    debt_over_time.append(final_state["debt"])
    savings_over_time.append(final_state["savings"])

# Print final statistics
print(f"Average Reward per Episode: {np.mean(cumulative_rewards)}")
print(f"Final Debt after {NUM_EPISODES} Episodes: {debt_over_time[-1]}")
print(f"Final Savings after {NUM_EPISODES} Episodes: {savings_over_time[-1]}")
print(f"Average Happiness Factor per Episode: {np.mean(happiness_factors)}")
