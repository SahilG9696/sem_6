import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
print(df)

print(df.head())
print(df.tail())
print(df.info)
print(df.describe())
print(df.tail())
print(df.shape)
print(df.dtypes)

# histogram
df.hist()
plt.show()

#box plot
df.boxplot()
plt.show()

# Histogram
sns.histplot(x = df['sepal_length'], kde=True)
plt.show()

sns.histplot(x = df['sepal_width'], kde=True)
plt.show()

sns.histplot(x = df['petal_length'], kde=True)
plt.show()

sns.histplot(x = df['petal_width'], kde=True)
plt.show()

# comparison of iris dataset features using Histogram
# Create a 2×2 grid for the histogram
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))  # Adjust size as needed

sns.histplot(x=df['sepal_length'], ax=axes[0, 0])
axes[0, 0].set_title("Plote_1:-Sepal Length")

sns.histplot(x=df['sepal_width'], ax=axes[0, 1])
axes[0, 1].set_title("Plote_2:-Sepal Width")

sns.histplot(x=df['petal_length'], ax=axes[1, 0])
axes[1, 0].set_title("Plote_3:-Petal Length")

sns.histplot(x=df['petal_width'], ax=axes[1, 1])
axes[1, 1].set_title("Plote_4:-Petal Width")

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


# 2×2 grid for the boxplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))  # Adjust size as needed

sns.histplot(x='petal_length', y='species', hue='species', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Plote_1:-Petal Length vs Species")

sns.histplot(x='sepal_length', y='species', hue='species', data=df, ax=axes[0, 1])
axes[0, 1].set_title("Plote_2:-Sepal Length vs Species")

sns.histplot(x='petal_width', y='species', hue='species', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Plote_3:-Petal Width vs Species")

sns.histplot(x='sepal_width', y='species', hue='species', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Plote_4:-Sepal Width vs Species")

# Adjust layout
plt.tight_layout()
plt.show()


# Box plot
sns.boxplot(df['sepal_length'])
plt.show()

sns.boxplot(df['sepal_width'])
plt.show()

sns.boxplot(df['petal_length'])
plt.show()

sns.boxplot(df['petal_width'])
plt.show()

# Create a 2×2 grid for the boxplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))  # Adjust size as needed

sns.boxplot(x=df['sepal_length'], ax=axes[0, 0])
axes[0, 0].set_title("Plote_1:-Sepal Length")

sns.boxplot(x=df['sepal_width'], ax=axes[0, 1])
axes[0, 1].set_title("Plote_2:-Sepal Width")

sns.boxplot(x=df['petal_length'], ax=axes[1, 0])
axes[1, 0].set_title("Plote_3:-Petal Length")

sns.boxplot(x=df['petal_width'], ax=axes[1, 1])
axes[1, 1].set_title("Plote_4:-Petal Width")

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


sns.boxplot(x='petal_length',y='species',hue = 'species',data=df)
plt.show()

sns.boxplot(x='sepal_length',y='species',hue='species',data=df)
plt.show()

sns.boxplot(x='petal_width',y='species',hue = 'species',data=df)
plt.show()

sns.boxplot(x='sepal_width',y='species',hue='species',data=df)
plt.show()


# 2×2 grid for the boxplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))  # Adjust size as needed

sns.boxplot(x='petal_length', y='species', hue='species', data=df, ax=axes[0, 0])
axes[0, 0].set_title("Plote_1:-Petal Length vs Species")

sns.boxplot(x='sepal_length', y='species', hue='species', data=df, ax=axes[0, 1])
axes[0, 1].set_title("Plote_2:-Sepal Length vs Species")

sns.boxplot(x='petal_width', y='species', hue='species', data=df, ax=axes[1, 0])
axes[1, 0].set_title("Plote_3:-Petal Width vs Species")

sns.boxplot(x='sepal_width', y='species', hue='species', data=df, ax=axes[1, 1])
axes[1, 1].set_title("Plote_4:-Sepal Width vs Species")

# Adjust layout
plt.tight_layout()
plt.show()






# # scatter plot
# plt.scatter(df["sepal_length"],df["sepal_width"])
# plt.xlabel('sepal_length')
# plt.ylabel('sepal_width')
# plt.show()
#
# #
# plt.scatter(df["sepal_length"],df["petal_length"])
# plt.xlabel('sepal_length')
# plt.ylabel('petal_length')
# plt.show()
#
# plt.scatter(df["sepal_length"],df["petal_width"])
# plt.xlabel('sepal_length')
# plt.ylabel('petal_width')
# plt.show()
#
# plt.scatter(df["sepal_width"],df["sepal_length"])
# plt.xlabel('sepal_width')
# plt.ylabel('sepal_length')
# plt.show()
#
# plt.scatter(df["sepal_width"],df["petal_length"])
# plt.xlabel('sepal_width')
# plt.ylabel('petal_length')
# plt.show()
#
# plt.scatter(df["sepal_width"],df["petal_width"])
# plt.xlabel('sepal_width')
# plt.ylabel('petal_width')
# plt.show()
#
# plt.scatter(df["petal_length"],df["sepal_length"])
# plt.xlabel('petal_length')
# plt.ylabel('sepal_length')
# plt.show()
#
# plt.scatter(df["petal_length"],df["sepal_width"])
# plt.xlabel('petal_length')
# plt.ylabel('sepal_width')
# plt.show()
#
# plt.scatter(df["petal_length"],df["petal_width"])
# plt.xlabel('petal_length')
# plt.ylabel("petal_width")
# plt.show()
#
# plt.scatter(df["petal_width"],df["sepal_length"])
# plt.xlabel('petal_width')
# plt.ylabel('sepal_length')
# plt.show()
#
# plt.scatter(df["petal_width"],df["sepal_width"])
# plt.xlabel('petal_width')
# plt.ylabel('sepal_width')
# plt.show()
#
# plt.scatter(df["petal_width"],df["petal_length"])
# plt.xlabel('petal_width')
# plt.ylabel('petal_length')
# plt.show()