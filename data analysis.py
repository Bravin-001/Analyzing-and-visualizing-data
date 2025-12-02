import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
df.head()

df.info()
df.isnull().sum()

# Clean missing values (if any)
df = df.dropna()

#group by species
species_means = df.groupby('species').mean()
species_means

#line chart
plt.figure(figsize=(8,4))
plt.plot(df.index, df['sepal length (cm)'])
plt.title("Sepal Length Trend Over Sample Index")
plt.xlabel("Sample Index (Time Proxy)")
plt.ylabel("Sepal Length (cm)")
plt.show()

#bar chart
plt.figure(figsize=(8,4))
species_means['petal length (cm)'].plot(kind='bar')
plt.title("Average Petal Length per Species")
plt.ylabel("Petal Length (cm)")
plt.xlabel("Species")
plt.show()

#histogram
plt.figure(figsize=(8,4))
plt.hist(df['sepal width (cm)'], bins=20)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

#scatter plot
plt.figure(figsize=(8,4))
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

