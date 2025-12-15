# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Iris dataset and explore its structure  ---
# We use seaborn to load the built-in iris dataset easily
print("Loading Dataset...")
df = sns.load_dataset('iris')

# Display the first 5 rows to understand structure
print("\n--- First 5 Rows ---")
print(df.head())

# Display the structure (columns and data types)
print("\n--- Dataset Info ---")
print(df.info())

# Display statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# --- 2. Check for missing values and handle them  ---
print("\n--- Checking for Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values)

# (Note: The Iris dataset is usually clean, but if there were missing values, 
# we would drop them using df.dropna() or fill them using df.fillna())
if df.isnull().values.any():
    print("\nHandling missing values...")
    df = df.dropna()
else:
    print("\nNo missing values found.")

# --- 3. Visualize the data  ---

# A. Pairplot to see relations between all features and species
print("\nGenerating Pairplot...")
plt.figure(figsize=(10, 6))
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# B. Boxplot to see distribution of attributes per species
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.boxplot(x='species', y='sepal_length', data=df)
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
sns.boxplot(x='species', y='sepal_width', data=df)
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
sns.boxplot(x='species', y='petal_length', data=df)
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
sns.boxplot(x='species', y='petal_width', data=df)
plt.title('Petal Width Distribution')

plt.tight_layout()
plt.show()

# C. Histogram to see frequency distribution of Sepal Length
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal_length'], kde=True, color='blue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()