# Day 5 Exercise Solutions

This directory contains the solutions for all Day 5 exercises on Data Manipulation and Analysis.

## Exercise Solutions

### Part 1: Data Loading and Exploration
- **[exercise_1_1.py](exercise_1_1.py)** - Loading and Inspecting Data
- **[exercise_1_2.py](exercise_1_2.py)** - Data Quality Assessment

### Part 2: Data Selection and Filtering
- **[exercise_2_1.py](exercise_2_1.py)** - Boolean Indexing
- **[exercise_2_2.py](exercise_2_2.py)** - .loc[] and .iloc[] Practice

### Part 3: Data Cleaning
- **[exercise_3_1.py](exercise_3_1.py)** - Handling Missing Data
- **[exercise_3_2.py](exercise_3_2.py)** - Detecting and Handling Outliers

### Part 4: Data Transformation and Grouping
- **[exercise_4_1.py](exercise_4_1.py)** - Creating New Variables
- **[exercise_4_2.py](exercise_4_2.py)** - Grouping and Aggregation

### Part 5: Reshaping and Combining Data
- **[exercise_5_1.py](exercise_5_1.py)** - Pivot Tables and Melting
- **[exercise_5_2.py](exercise_5_2.py)** - Merging and Concatenating

### Part 6: Statistical Analysis with SciPy
- **[exercise_6_1.py](exercise_6_1.py)** - Hypothesis Testing
- **[exercise_6_2.py](exercise_6_2.py)** - Correlation and Distribution Fitting

### Challenge Exercises
- **[challenge_1.py](challenge_1.py)** - Complete Data Analysis Pipeline
- **[challenge_2.py](challenge_2.py)** - Advanced Grouping and Aggregation
- **[challenge_3.py](challenge_3.py)** - Missing Data Analysis

## How to Use These Solutions

1. **Try the exercises first** - Always attempt to solve the exercises on your own before looking at the solutions
2. **Compare approaches** - After solving, compare your solution with the provided one
3. **Understand, don't copy** - Make sure you understand why each step is taken
4. **Experiment** - Modify the solutions to explore different approaches

## Running the Solutions

Each solution file assumes you have already created the sample dataset as described in Exercise 1.1:

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_subjects = 30
n_trials_per_condition = 10

data = {
    'subject_id': np.repeat(range(1, n_subjects + 1), n_trials_per_condition * 3),
    'condition': np.tile(['control', 'drug_a', 'drug_b'], n_subjects * n_trials_per_condition),
    'trial': np.tile(range(1, n_trials_per_condition + 1), n_subjects * 3),
    'reaction_time': np.random.normal(350, 80, n_subjects * n_trials_per_condition * 3),
    'accuracy': np.random.uniform(0.6, 1.0, n_subjects * n_trials_per_condition * 3),
    'session': np.random.choice(['morning', 'afternoon'], n_subjects * n_trials_per_condition * 3)
}

df = pd.DataFrame(data)

# Add some realistic effects
df.loc[df['condition'] == 'drug_a', 'reaction_time'] += 30
df.loc[df['condition'] == 'drug_b', 'reaction_time'] -= 20
df.loc[df['session'] == 'afternoon', 'reaction_time'] += 15

# Round values
df['reaction_time'] = df['reaction_time'].round(1)
df['accuracy'] = df['accuracy'].round(3)
```

Then you can run any solution file directly or copy the code into your Jupyter notebook.

## Tips for Learning

- **Work through exercises sequentially** - Each builds on previous concepts
- **Experiment with variations** - Try different parameters and approaches
- **Read the comments** - Solutions include explanatory comments
- **Check the output** - Make sure you understand what each operation produces
- **Ask questions** - If something is unclear, ask your instructor or peers

