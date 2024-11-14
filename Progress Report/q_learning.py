import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import csv

# Constants and Hyperparameters
NUM_MONTHS = 12
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON_INITIAL = 0.3
EPSILON_DECAY = 0.99

# Financial parameters
SAVINGS_GROWTH_RATE = 0.005
DEBT_GROWTH_RATE = 0.01
PROMOTION_PROB = 0.10
LAYOFF_PROB = 0.05
MEDICAL_EXPENSE_PROB = 0.02
HOME_REPAIR_PROB = 0.03
MEDICAL_COST = 5000
HOME_REPAIR_COST = 2000
PROMOTION_INCREASE = 1.15
LAYOFF_DECREASE = 0.5

# Initial conditions
initial_state = {"savings": 10000, "debt": 5000, "income": 4000, "expenses": 2000}

actions = ["allocate_savings", "pay_debt", "increase_spending"]
Q = {}


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

    if action == "allocate_savings":
        state["savings"] += state["income"] - state["expenses"]
    elif action == "pay_debt":
        payment = min(state["debt"], state["income"] - state["expenses"])
        state["debt"] -= payment
        state["savings"] += state["income"] - state["expenses"] - payment
    elif action == "increase_spending":
        state["expenses"] *= 1.05

    return state


# Reward function
def reward(state):
    debt_penalty = -0.2 * state["debt"]
    savings_reward = 0.1 * state["savings"]
    total_reward = savings_reward + debt_penalty
    return total_reward


# Epsilon-greedy policy
def choose_action(state, epsilon):
    state_tuple = tuple(state.values())
    if random.random() < epsilon or state_tuple not in Q:
        return random.choice(actions)
    else:
        return max(Q[state_tuple], key=Q[state_tuple].get)


# Q-learning update function
def update_q_table(prev_state, action, reward, next_state):
    prev_state_tuple = tuple(prev_state.values())
    next_state_tuple = tuple(next_state.values())

    if prev_state_tuple not in Q:
        Q[prev_state_tuple] = {a: 0.0 for a in actions}
    if next_state_tuple not in Q:
        Q[next_state_tuple] = {a: 0.0 for a in actions}

    best_next_action = max(Q[next_state_tuple].values())
    Q[prev_state_tuple][action] = round(
        (1 - LEARNING_RATE) * Q[prev_state_tuple][action]
        + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next_action),
        2,
    )


# Simulation function
def simulate_q_learning(num_months, epsilon):
    current_state = initial_state.copy()
    total_reward = 0

    for month in range(num_months):
        action = choose_action(current_state, epsilon)
        prev_state = current_state.copy()
        current_state = transition(current_state, action)
        monthly_reward = reward(current_state)
        update_q_table(prev_state, action, monthly_reward, current_state)
        total_reward += monthly_reward

    return round(total_reward, 2), current_state


# Evaluation
NUM_EPISODES = 10
cumulative_rewards = []
debt_over_time = []
savings_over_time = []
q_value_changes = []
prev_Q = {}

for episode in range(NUM_EPISODES):
    epsilon = max(0.1, EPSILON_INITIAL * (EPSILON_DECAY**episode))
    total_reward, final_state = simulate_q_learning(NUM_MONTHS, epsilon)

    cumulative_rewards.append(total_reward)
    debt_over_time.append(round(final_state["debt"], 2))
    savings_over_time.append(round(final_state["savings"], 2))

    q_change = 0
    for state_tuple in Q:
        for action in Q[state_tuple]:
            if state_tuple in prev_Q and action in prev_Q[state_tuple]:
                q_change += abs(Q[state_tuple][action] - prev_Q[state_tuple][action])
    q_value_changes.append(round(q_change, 2))
    prev_Q = copy.deepcopy(Q)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode (Reward Analysis)")
plt.savefig("cumulative_rewards.png")

plt.figure(figsize=(10, 6))
plt.plot(debt_over_time, label="Debt")
plt.plot(savings_over_time, label="Savings")
plt.xlabel("Episode")
plt.ylabel("Balance")
plt.title("Debt and Savings Over Episodes (Financial Stability)")
plt.legend()
plt.savefig("q_learning_debt_savings_over_time.png")

plt.figure(figsize=(10, 6))
plt.plot(q_value_changes)
plt.xlabel("Episode")
plt.ylabel("Q-value Change")
plt.title("Q-value Changes Over Episodes (Policy Convergence)")
plt.savefig("q_learning_q_value_changes.png")

# Save Q-values to CSV
with open("q_learning_q_values.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["State", "Action", "Q-Value"])
    for state_tuple, actions_dict in Q.items():
        for action, q_value in actions_dict.items():
            writer.writerow([state_tuple, action, round(q_value, 2)])

# Print final statistics
average_reward = round(np.mean(cumulative_rewards), 2)
final_debt = round(debt_over_time[-1], 2)
final_savings = round(savings_over_time[-1], 2)
total_q_change = round(q_value_changes[-1], 2)

print(f"Average Reward per Episode: {average_reward:.2f}")
print(f"Final Debt after {NUM_EPISODES} Episodes: {final_debt:.2f}")
print(f"Final Savings after {NUM_EPISODES} Episodes: {final_savings:.2f}")
print(f"Total Q-value Change in Final Episode: {total_q_change:.2f}")
