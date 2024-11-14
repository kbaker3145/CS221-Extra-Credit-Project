import subprocess
import re
import csv

# Define the script names
scripts = ["random_action.py", "q_learning.py"]

# Define the number of trials
num_trials = 10


# Function to extract the result metrics from script output
def extract_metrics(output):
    # Use regular expressions to extract metrics from the output text
    avg_reward = float(
        re.search(r"Average Reward per Episode: ([\d\.]+)", output).group(1)
    )
    final_debt = float(
        re.search(r"Final Debt after \d+ Episodes: ([\d\.]+)", output).group(1)
    )
    final_savings = float(
        re.search(r"Final Savings after \d+ Episodes: ([\d\.]+)", output).group(1)
    )
    return avg_reward, final_debt, final_savings


# Function to run each script multiple times and calculate averages
def run_trials(script):
    avg_rewards = []  # List to store average rewards
    final_debts = []  # List to store final debts
    final_savings = []  # List to store final savings

    for _ in range(num_trials):
        # Run the script and capture the output
        result = subprocess.run(["python", script], capture_output=True, text=True)
        output = result.stdout

        # Extract metrics from the output
        avg_reward, final_debt, final_savings_trial = extract_metrics(output)

        # Append each metric to the corresponding list
        avg_rewards.append(avg_reward)
        final_debts.append(final_debt)
        final_savings.append(final_savings_trial)

    # Calculate average metrics across all trials and round to two decimal places
    overall_avg_reward = round(sum(avg_rewards) / num_trials, 2)
    overall_final_debt = round(sum(final_debts) / num_trials, 2)
    overall_final_savings = round(sum(final_savings) / num_trials, 2)

    return overall_avg_reward, overall_final_debt, overall_final_savings


# Run evaluations for each script and save results to CSV
with open("trial_results.csv", mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write header
    writer.writerow(
        [
            "Script",
            "Average Reward per Episode",
            "Final Debt after Trials",
            "Final Savings after Trials",
        ]
    )

    # Run each script and record results
    for script in scripts:
        print(f"Evaluating {script} over {num_trials} trials...")
        avg_reward, final_debt, final_savings = run_trials(script)

        # Print results to console, rounded to two decimal places
        print(f"Overall Average Reward per Episode: {avg_reward:.2f}")
        print(f"Overall Final Debt after {num_trials} Episodes: {final_debt:.2f}")
        print(f"Overall Final Savings after {num_trials} Episodes: {final_savings:.2f}")
        print("-" * 50)

        # Write results to CSV, rounded to two decimal places
        writer.writerow(
            [script, f"{avg_reward:.2f}", f"{final_debt:.2f}", f"{final_savings:.2f}"]
        )

print("Results saved to trial_results.csv")
