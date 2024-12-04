import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the script names
scripts = ["optimized_q_learning.py", "random_action.py"]

# Define the number of trials
num_trials = 10

# Define categorized input combinations
input_combinations = {
    "Median Household": {"income": 6215, "expenses": 5577, "savings": 5300, "debt": 6380},
    "Struggling Household": {"income": 4500, "expenses": 4000, "savings": 1000, "debt": 12000},
    "Slightly Above Median Household": {"income": 6500, "expenses": 5500, "savings": 15000, "debt": 7000},
    "Well-Off Household": {"income": 9000, "expenses": 7500, "savings": 25000, "debt": 1000},
}

# Function to extract the result metrics
def extract_metrics(output):
    avg_reward_match = re.search(r"Average Reward per Episode: ([\-?\d\.]+)", output)
    final_debt_match = re.search(r"Final Debt after \d+ Episodes: ([\-?\d\.]+)", output)
    final_savings_match = re.search(r"Final Savings after \d+ Episodes: ([\-?\d\.]+)", output)
    avg_happiness_match = re.search(r"Average Happiness Factor per Episode: ([\-?\d\.]+)", output)
    
    if not (avg_reward_match and final_debt_match and final_savings_match and avg_happiness_match):
        print(f"Full script output:\n{output}")
        raise ValueError(f"Output does not contain the required metrics:\n{output}")
    
    avg_reward = float(avg_reward_match.group(1))
    final_debt = float(final_debt_match.group(1))
    final_savings = float(final_savings_match.group(1))
    avg_happiness = float(avg_happiness_match.group(1))
    return avg_reward, final_debt, final_savings, avg_happiness

# Function to run each script multiple times and calculate averages
def run_trials(script, initial_state):
    avg_rewards = []
    final_debts = []
    final_savings = []
    avg_happiness_factors = []
    
    for _ in range(num_trials):
        result = subprocess.run(
            [
                "python", script,
                f"--income={initial_state['income']}",
                f"--debt={initial_state['debt']}",
                f"--savings={initial_state['savings']}",
                f"--expenses={initial_state['expenses']}"
            ],
            capture_output=True, text=True
        )
        output = result.stdout
        avg_reward, final_debt, final_savings_trial, avg_happiness = extract_metrics(output)
        
        avg_rewards.append(avg_reward)
        final_debts.append(final_debt)
        final_savings.append(final_savings_trial)
        avg_happiness_factors.append(avg_happiness)
    
    return {
        "avg_reward": np.mean(avg_rewards),
        "final_debt": np.mean(final_debts),
        "final_savings": np.mean(final_savings),
        "avg_happiness": np.mean(avg_happiness_factors),
        "initial_happiness": 0.05 * initial_state["expenses"],
    }

# Collect results for visualization
results = []

for script in scripts:
    for category, initial_state in input_combinations.items():
        print(f"Evaluating {script} with category {category} over {num_trials} trials...")
        metrics = run_trials(script, initial_state)
        metrics.update({"Script": script, "Category": category})
        results.append(metrics)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Visualization of Bar Graphs with Baseline Happiness Line
def plot_bar_with_baseline_line(metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = list(input_combinations.keys())
    x = np.arange(len(categories))
    width = 0.35  # Bar width

    # Plot bars
    for i, script in enumerate(scripts):
        subset = results_df[results_df["Script"] == script]
        means = subset[metric]
        ax.bar(
            x + i * width - width / 2,
            means,
            width,
            label=script
        )

    # Calculate and plot the baseline happiness factor as a dotted line
    baseline_happiness_values = [input_combinations[category]["expenses"] * 0.05 * 60 for category in categories]
    avg_baseline_happiness = np.mean(baseline_happiness_values)
    ax.hlines(
        y=avg_baseline_happiness,
        xmin=-0.5,  # Extend to include the entire plot area
        xmax=len(categories) - 0.5,
        colors="black",
        linestyles="dotted",
        label="Baseline Happiness Factor"
    )

    ax.set_xlabel("Household Category")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Create bar graph with baseline happiness line
plot_bar_with_baseline_line(
    metric="avg_happiness",
    ylabel="Average Happiness Factor",
    title="Happiness Factor Comparison: Random Actions vs. Q-Learning",
    filename="happiness_factor_comparison.png"
)

# Create bar graphs for final savings and debt
def plot_bar(metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    categories = list(input_combinations.keys())
    x = np.arange(len(categories))
    width = 0.35  # Bar width

    for i, script in enumerate(scripts):
        subset = results_df[results_df["Script"] == script]
        means = subset[metric]
        ax.bar(
            x + i * width - width / 2,
            means,
            width,
            label=script
        )

    ax.set_xlabel("Household Category")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_bar(
    metric="final_savings",
    ylabel="Final Savings",
    title="Final Savings Comparison: Random Actions vs. Q-Learning",
    filename="final_savings_comparison.png"
)
plot_bar(
    metric="final_debt",
    ylabel="Final Debt",
    title="Final Debt Comparison: Random Actions vs. Q-Learning",
    filename="final_debt_comparison.png"
)

print("Results visualized and saved as bar graphs.")
