"""
Week 1 Task: Pandas & NumPy Data Analysis
Student: Test Submission
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and explore a sample dataset
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "age":     np.random.randint(18, 65, n),
    "salary":  np.random.normal(50000, 15000, n).round(2),
    "score":   np.random.uniform(0, 100, n).round(1),
    "dept":    np.random.choice(["Engineering", "Marketing", "Finance", "HR"], n),
})

# Basic exploration
print("=== Dataset Overview ===")
print(data.head())
print("\nShape:", data.shape)
print("\nDescriptive Stats:")
print(data.describe())

# Group analysis
dept_avg = data.groupby("dept")["salary"].mean().sort_values(ascending=False)
print("\nAverage Salary by Department:")
print(dept_avg)

# Correlation
numeric_cols = data.select_dtypes(include=np.number)
corr = numeric_cols.corr()
print("\nCorrelation Matrix:")
print(corr)

# Filter high earners
high_earners = data[data["salary"] > data["salary"].mean() + data["salary"].std()]
print(f"\nHigh earners ({len(high_earners)} people): avg score = {high_earners['score'].mean():.2f}")
