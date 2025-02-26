import pandas as pd
import numpy as np

# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for
# a dataset (age, income etc.) with numeric variables grouped by one of the qualitative
# (categorical) variable. For example, if your categorical variable is age groups and
# quantitative variable is income, then provide summary statistics of income grouped by the
# age groups. Create a list that contains a numeric value for each response to the categorical
# variable.

""" 
list_1 = {"Age_Group":["Young","Young","Middle Aged","Middle Aged","Senior","Senior"],
          "Income":[30000,32000,50000,52000,32000,37000]
          }

# Converting Dictionary into dataset
df = pd.DataFrame(list_1)

# Display the first few rows
print(df)

# Grouping by 'Age Group' and calculating summary statistics
summary_stats = df.groupby("Age_Group")["Income"].agg(["mean", "median", "min", "max", "std"])
print(summary_stats)
"""

# 2. Write a Python program to display some basic statistical details like percentile, mean,
# standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’
# of iris.csv dataset.

# sepal is a green part of flower
# petal is a colorful part of flower
# sepal length means height of green part
# sepal width means width of green part
# petal length means height of colourful part
# petal width means width of colourful part


# Load the iris dataset
df_iris = pd.read_csv("iris.csv")

# Display first few rows
print(df_iris)

print(df_iris.describe())
print(df_iris.info)

print(df_iris.loc[:,'sepal_length'].mean())
print(df_iris.loc[:,'sepal_width'].mean())
print(df_iris.loc[:,'petal_length'].mean())
print(df_iris.loc[:,'sepal_width'].mean())

print(df_iris.loc[:,'sepal_length'].mode())
print(df_iris.loc[:,'sepal_width'].mode())
print(df_iris.loc[:,'petal_length'].mode())
print(df_iris.loc[:,'sepal_width'].mode())

print(df_iris.loc[:,'sepal_length'].median())
print(df_iris.loc[:,'sepal_width'].median())
print(df_iris.loc[:,'petal_length'].median())
print(df_iris.loc[:,'sepal_width'].median())

print(df_iris.loc[:,'sepal_length'].std())
print(df_iris.loc[:,'sepal_width'].std())
print(df_iris.loc[:,'petal_length'].std())
print(df_iris.loc[:,'sepal_width'].std())

# Filter data for each species and calculate statistics
summary_iris = df_iris.groupby("species").agg(["mean", "std", "min", "max", "median"])
print(summary_iris)




