"""
File Name: Epsilon-Greedy Q-Learning

This script implements an optimized Q-Learning algorithm using epsilon-greedy exploration
with features such as reward shaping, improved state representation, extended exploration,
and multi-objective rewards.
"""

import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import copy

# Constants and Hyperparameters
NUM_MONTHS = 60  
NUM_EPISODES = 100 
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.85
EPSILON_INITIAL = 0.3
EPSILON_DECAY = 0.99  

# Financial parameters
SAVINGS_GROWTH_RATE = 0.005  # 0.5% savings growth rate
DEBT_GROWTH_RATE = 0.01  # 1% debt growth rate
PROMOTION_PROB = 0.10  # 10% chance for promotion
LAYOFF_PROB = 0.05  # 5% chance of layoff
MEDICAL_EXPENSE_PROB = 0.02  # 2% chance of medical emergency
HOME_REPAIR_PROB = 0.03  # 3% chance of home repair
MEDICAL_COST = 1000
HOME_REPAIR_COST = 200
PROMOTION_INCREASE = 1.15
LAYOFF_DECREASE = 0.8

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Optimized Q-Learning Budget Optimization")
parser.add_argument("--income", type=float, required=True, help="Initial monthly income")
parser.add_argument("--expenses", type=float, required=True, help="Initial monthly expenses")
parser.add_argument("--savings", type=float, required=True, help="Initial savings")
parser.add_argument("--debt", type=float, required=True, help="Initial debt")
args = parser.parse_args()

# Initial conditions
initial_state = {
    "income": args.income,
    "expenses": args.expenses,
    "savings": args.savings,
    "debt": args.debt
}

actions = ["allocate_savings", "pay_debt", "increase_spending"]
Q = {}  # Q-table

# Transition function
def transition(state, action):
    state = state.copy()
    state["savings"] += state["savings"] * SAVINGS_GROWTH_RATE
    state["debt"] += state["debt"] * DEBT_GROWTH_RATE

    income_event = random.random()
    if income_event < PROMOTION_PROB:
        state["income"] *= PROMOTION_INCREASE
    elif income_event < (PROMOTION_PROB + LAYOFF_PROB):
        state["income"] *= LAYOFF_DECREASE

    if random.random() < MEDICAL_EXPENSE_PROB:
        state["savings"] -= MEDICAL_COST
    if random.random() < HOME_REPAIR_PROB:
        state["savings"] -= HOME_REPAIR_COST

    state["disposable_income"] = state["income"] - state["expenses"]
    if state["disposable_income"] < 0:
        state["debt"] -= state["disposable_income"]
    else:
        if action == "allocate_savings":
            state["savings"] += state["disposable_income"]
        elif action == "pay_debt":
            payment = min(abs(state["debt"]), state["disposable_income"])
            state["debt"] -= payment
            state["savings"] += (state["disposable_income"] - payment)
        elif action == "increase_spending" and state["savings"] > state["income"] * 0.05 * NUM_MONTHS:
            state["expenses"] *= 1.05

    return state

def reward(state):
    savings_reward = 5000 * state["savings"]
    debt_penalty = -1000 * state["debt"]
    happiness_reward = 0.05 * state["expenses"]

    # Additional rewards for milestones
    intermediate_reward = 0
    if state["savings"] > 10000:
        intermediate_reward += 2000
    elif state["savings"] > 20000:
        intermediate_reward += 5000

    # Total reward calculation
    total_reward = (savings_reward + debt_penalty + happiness_reward +
                    intermediate_reward)
    
    return total_reward, happiness_reward

# Epsilon-greedy action selection for exploration
def epsilon_greedy_action_selection(state, epsilon):
    state_tuple = tuple(state.values())
    if state_tuple not in Q or random.random() < epsilon:
        return random.choice(actions)
    return max(Q[state_tuple], key=Q[state_tuple].get)

# Update Q-table
def update_q_table(prev_state, action, reward, next_state):
    prev_state_tuple = tuple(prev_state.values())
    next_state_tuple = tuple(next_state.values())

    if prev_state_tuple not in Q:
        Q[prev_state_tuple] = {a: 0.0 for a in actions}
    if next_state_tuple not in Q:
        Q[next_state_tuple] = {a: 0.0 for a in actions}

    best_next_action = max(Q[next_state_tuple].values())
    Q[prev_state_tuple][action] = ((1 - LEARNING_RATE) * Q[prev_state_tuple][action] +
                                   LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next_action))

# Simulate Q-learning
cumulative_rewards = []
happiness_factors = []
debt_over_time = []
savings_over_time = []

for episode in range(NUM_EPISODES):
    epsilon = max(0.1, EPSILON_INITIAL * (EPSILON_DECAY ** episode))
    current_state = initial_state.copy()
    total_reward = 0
    total_happiness = 0

    for month in range(NUM_MONTHS):
        action = epsilon_greedy_action_selection(current_state, epsilon)
        prev_state = current_state.copy()
        current_state = transition(current_state, action)
        monthly_reward, happiness_reward = reward(current_state)
        update_q_table(prev_state, action, monthly_reward, current_state)
        total_reward += monthly_reward
        total_happiness += happiness_reward

    cumulative_rewards.append(total_reward)
    happiness_factors.append(total_happiness)
    debt_over_time.append(current_state["debt"])
    savings_over_time.append(current_state["savings"])

# Print final statistics
print(f"Average Reward per Episode: {np.mean(cumulative_rewards)}")
print(f"Final Debt after {NUM_EPISODES} Episodes: {debt_over_time[-1]}")
print(f"Final Savings after {NUM_EPISODES} Episodes: {savings_over_time[-1]}")
print(f"Average Happiness Factor per Episode: {np.mean(happiness_factors)}")
