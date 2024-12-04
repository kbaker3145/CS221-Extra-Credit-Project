**General structure of scripts**

---

### **Uncertainty Modeling**

Uncertainty was modeled for random\_actions.py and q\_learning.py using probabilities to simulate real-world unpredictability:

* **Income Fluctuations**:  
  * 10% chance of promotion (`income * 1.15`).  
  * 5% chance of layoff (`income * 0.8`).  
* **Unexpected Expenses**:  
  * 2% chance of medical expenses ($1,000).  
  * 3% chance of home repair expenses ($200).  
* **Savings and Debt Growth**:  
  * Monthly savings growth: 0.5%.  
  * Monthly debt growth: 1%.

---

### **Input Customization**

Both `random_action.py` and `q_learning.py` were updated to accept **initial state values** (e.g., income, savings, debt) directly from the terminal (our end-to-end implementation). This allows users to simulate various financial scenarios using commands like:

python random\_actions.py \--income=6215 \--expenses=5577 \--savings=5300 \--debt=6380  
python q\_learning.py \--income=6215 \--expenses=5577 \--savings=5300 \--debt=6380  
---

**q\_learning.py**

### **Comprehensive Write-Up: Optimized Q-Learning for Budget Optimization**

This script implements an advanced Q-Learning algorithm tailored for household budget optimization, incorporating nuanced financial dynamics and multi-objective rewards. Below, we outline the critical components and improvements made to the original implementation:

---

### **Objectives**

The algorithm optimizes not only for **increased savings** and **reduced debt** but also includes a **happiness factor** that incentivizes spending. This modification ensures that the policy reflects realistic human behavior, balancing financial goals with quality-of-life considerations:

* **Happiness Factor**: Integrated into the reward function, it grows as expenses increase, up to a reasonable cap.

---

### **Budgeting and Learning**

Initially, the Q-Learning implementation set 10 episodes of 12 months, mistakenly equating episodes with budgeting years. This was corrected:

* **Episodes**: Represent learning iterations, where the model refines its policy with each episode.  
* **Months**: Represent the actual budgeting period. For a 5-year budget, `NUM_MONTHS` was set to 60\.  
* **Improved Learning**: Increased `NUM_EPISODES` to 100, allowing the model to explore and converge on the optimal policy.

---

### **State Transition Adjustments**

The way state values (e.g., savings, debt) change based on actions was refined for realism:

1. **Disposable Income**: Calculated as `income - expenses`. If negative, it directly increases debt.  
2. **Actions**:  
   * **allocate\_savings**: Allocates disposable income to savings.  
   * **pay\_debt**: Prioritizes debt repayment, with any remainder added to savings.  
   * **increase\_spending**: Allowed only if savings exceed a threshold (`income * 0.05 * NUM_MONTHS`).

---

### **Optimizations in Q-Learning**

#### **Reward Function**

* **Original**: Basic rewards for savings and happiness; penalties for debt.  
* **Optimized**:  
  * **Extreme Penalties**: Increased negative weight for debt to prioritize repayment.  
  * **Savings Milestones**: Intermediate rewards for reaching savings targets:  
    * \+$2,000 for savings \> $10,000.  
    * \+$5,000 for savings \> $20,000.

#### **Hyperparameters**

* Adjusted for improved learning dynamics:  
  * **Learning Rate**: Increased from 0.1 to 0.2 for faster updates.  
  * **Discount Factor**: Reduced from 0.9 to 0.85 to balance short- and long-term rewards.  
  * **Epsilon Decay**: Slowed (0.99) for extended exploration.

#### **Action Selection**

* Epsilon-greedy exploration was used (softmax action selection explored in experiment below)

**run\_trials.py**

### **Testing Multiple Scenarios**

The testing process in `run_trials.py` was expanded to evaluate the algorithms across 10 trials for diverse financial situations. Input values were derived from reliable sources and represent reasonable deviations from median statistics:

* **Income**: Median household income in 2022: $74,580 (U.S. Census Bureau [*U.S. Census Bureau*](https://www.census.gov/library/publications/2023/demo/p60-279.html?utm_source=chatgpt.com)).  
* **Expenses**: Average annual household expenditures in 2023: $66,928 (Bureau of Labor Statistics [*Bureau of Labor Statistics*](https://www.bls.gov/news.release/cesan.nr0.htm?utm_source=chatgpt.com)).  
* **Savings**: Median savings account balance: $5,300; Mean balance: $41,600 (Pew Research Center [*Pew Research Center*](https://www.pewresearch.org/2023/12/04/the-assets-households-own-and-the-debts-they-carry/?utm_source=chatgpt.com)).  
* **Debt**: Average credit card balance per borrower: $6,380 (The Fool [*The Fool*](https://www.fool.com/money/research/average-household-debt/?utm_source=chatgpt.com)).

Each input combination was categorized into distinct household types, such as "Median Household," "Struggling Household," and "Well-Off Household." `run_trials.py` generates comparison graphs for **savings**, **debt**, and **happiness factor**.

### **Graphical Comparisons**

The performance of `random_action.py` and `q_learning.py` was visualized in bar graphs with:

* **Metrics**: Final savings, final debt, and average happiness factor.  
* The **dotted line** represents the happiness level that would occur **without any optimization to spending**, i.e., if spending decisions were not tailored by the Q-learning algorithm and followed a random or static pattern instead.

**RESULTS:**

**Optimized q learning vs baseline:**

These are the results comparing the final optimized q learning to random actions (the baseline that takes random actions at each month). 

![][image1]![][image2]![][image3]

- Q\_learning always optimizes happiness except for struggling household, and is better than random for well-off households.

OTHER EXPERIMENTS RUN TO OPTIMIZE Q LEARNING TO WHAT IT IS NOW:

**Testing whether to use softmax action selection or epsilon greedy exploration:**

We considered leveraging softmax exploration for smoother exploration-exploitation trade-off. In these, actions are selected probabilistically based on their Q-values and a temperature parameter controls the randomness. 

The optimized q\_learning ended up using epsilon greedy selection because softmax action selection dramatically reduced savings while being comparable across debt and happiness.

!

**Testing how much to constrain spending increases by:**

We tested a high and low constraint on spending (savings must be above 30% of income vs. 5% of income to increase spending). 

Optimized q learning ended up using the low constraint 5%, resulting in better savings and happiness, with only one category worse in debt.


**Testing whether or not to include savings milestones as such:**

* **Savings Milestones**: Intermediate rewards for reaching savings targets:  
  * \+$2,000 for savings \> $10,000.  
    * \+$5,000 for savings \> $20,000.

We tried adding rewards for hitting savings milestones to bump up low savings. This improved performance dramatically.
