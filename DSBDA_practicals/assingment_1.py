import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("titanic.csv")

# Various functions of pandas to print the data in different forms
print(df.isnull().sum())  # Check missing values
print(df.info())  # Dataset info
print(df.describe())  # Statistical summary
print(df.head())  # First 5 rows
print(df.tail())  # Last 5 rows
print(df.shape)  # Shape of the dataset (rows, columns)
print(df.columns)  # Column names

# Converting a categorical variable (string-based) into a quantitative variable (integer-based)
# Method 1
print(df['sex'].replace(['male', 'female'], [0, 1]))  # Convert 'sex' column to numerical
print(df['embarked'].replace(['S', 'C', 'Q'], [0, 1, 2]))  # Convert 'embarked' column

# Method 2
df['Emb_cat'] = df['embarked'].astype('category')  # Convert 'embarked' to category
print(df.head())
print(df['Emb_cat'].cat.codes)  # Print numeric codes

# Replacing null values
# Method 1 - Replacing null values with a placeholder ('NA')
print(df.isnull().sum())  # Check missing values
print(df.isnull())  # Print boolean mask for missing values

# For the 'age' column

df['age'].fillna('NA', inplace=True)
print(df["age"])

# For the 'deck' column

df['deck'].fillna('NA', inplace=True)
print(df["deck"])

# For the 'embarked' column
df['embarked'].fillna('NA', inplace=True)
print(df["embarked"])

# Method 2 - Deleting rows containing null values
print(df.isnull().sum())  # Check again for missing values
print(df.dropna())  # Remove rows with null values
print(df.isnull().sum())  # Final check
