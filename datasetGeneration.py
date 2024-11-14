import pandas as pd
import numpy as np

# Define constants for the dataset generation without economic state influence
NUM_TIMESTEPS = 500 # Number of months (timestep length)
START_SAVINGS = 5000 # Initial savings balance
START_DEBT = 3000 # Initial debt balance
START_INCOME = 4000  # Initial monthly income
START_EXPENSES = 2500  # Initial essential expenses
ACTIONS = ["Allocate Funds", "Repay Debt", "Adjust Lifestyle"]

# Define probabilities for income fluctuations and unexpected expenses
PROMOTION_PROB = 0.1
LAYOFF_PROB = 0.05
MEDICAL_EMERGENCY_PROB = 0.02
HOME_REPAIR_PROB = 0.03


# Function to simulate income
def simulate_income(current_income):
    return current_income  # No changes based on economic state


# Function to simulate interest growth for savings
def simulate_savings_growth(savings, action):
    if action == "Allocate Funds":
        growth_rate = 0.04  # Fixed interest growth rate
        return savings * (1 + growth_rate)
    return savings


# Function to simulate debt repayment
def simulate_debt_change(debt, action):
    if action == "Repay Debt":
        return max(0, debt - 500)  # Pay off $500 of debt
    return debt


# Function to simulate unexpected expenses
def simulate_expenses(expenses):
    if np.random.rand() < MEDICAL_EMERGENCY_PROB:
        return expenses + 5000  # Medical emergency cost
    elif np.random.rand() < HOME_REPAIR_PROB:
        return expenses + 2000  # Home repair cost
    return expenses


# Generate the dataset without economic state influence
data = []
savings = START_SAVINGS
debt = START_DEBT
income = START_INCOME
expenses = START_EXPENSES

for month in range(1, NUM_TIMESTEPS + 1):
    # Determine action
    action = np.random.choice(ACTIONS)

    # Simulate income, savings growth, debt change, and expenses
    income = simulate_income(income)
    savings = simulate_savings_growth(savings, action)
    debt = simulate_debt_change(debt, action)
    expenses = simulate_expenses(expenses)

    # Apply income and expenses to update savings
    savings = savings + income - expenses

    # Check for layoffs or promotions
    if np.random.rand() < PROMOTION_PROB:
        income *= 1.15  # 15% salary increase
    elif np.random.rand() < LAYOFF_PROB:
        income *= 0.5  # 50% salary reduction

    # Append data for the current timestep
    data.append(
        {
            "Step": month,
            "Savings Balance": savings,
            "Debt Balance": debt,
            "Income": income,
            "Essential Expenses": expenses,
            "Action": action,
        }
    )

# Create DataFrame
dataframe = pd.DataFrame(data)

# Write the dataset to a CSV file
output_path = "./mdp_dataset_no_economic_state.csv"
dataframe.to_csv(output_path, index=False)

print(f"Data has been successfully written to: {output_path}")

