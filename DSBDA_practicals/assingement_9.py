import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')
pd.set_option('display.max_rows', None)  # None means no limit on rows
pd.set_option('display.max_columns', None)  # None means no limit on columns
print(titanic.head())


# 1]count plot
sns.countplot(x='sex',data=titanic,hue='survived')
plt.show()


# 2)Pie Chart
titanic['sex'].value_counts().plot(kind="pie", autopct="%.2f")
plt.show()

# Numerical data
# 1] Histogram
plt.hist(titanic['age'], bins=5)
plt.show()

# BOX PLOT
plt.figure(figsize=(10,6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)    # hue = survived parameter adds color coding to the plot,
plt.show()

# BAR PLOT
sns.barplot(x='sex', y='age', hue='survived', data=titanic)
plt.show()

# cross tab
pd.crosstab(titanic['pclass'],titanic['survived'])
plt.show()

# heatmap
sns.heatmap(pd.crosstab(titanic['pclass'],titanic['survived']))
plt.show()

#cluster tab
sns.clustermap(pd.crosstab(titanic['parch'],titanic['survived']))
plt.show()

# count plot
sns.countplot(titanic['survived'])
plt.show()


# Group the data by gender ('sex') and survival status ('survived')
grouped = titanic.groupby(['sex', 'survived'])['age'].describe()

# Display the summary statistics
print(grouped)


# Create a box plot
#The x-axis will show "male" and "female" (gender).
#The y-axis will show the ages of passengers.
#The colors will show whether each passenger survived or not (1 = survived, 0 = did not survive).

# # BOX PLOT
# plt.figure(figsize=(10,6))
# sns.boxplot(x='sex', y='age', hue='survived', data=titanic)    # hue = survived parameter adds color coding to the plot,
# plt.show()
#
# # SWARMPLOT
# sns.swarmplot(x='sex', y='age', hue='survived', data=titanic)
# plt.show()
#
# # VIOLINPLOT
# sns.violinplot(x='sex', y='age', hue='survived', data=titanic)
# plt.show()
#
# # HISTO GRAM
# sns.histplot(x='sex', y='age', hue='survived', data=titanic)
# plt.show()
#
# # RUG PLOT
# sns.rugplot(x='sex', y='age', hue='survived', data=titanic)
# plt.show()
#
# # BAR PLOT
# sns.barplot(x='sex', y='age', hue='survived', data=titanic)
# plt.show()
#
# # joint plot
# sns.jointplot(x=titanic['sex'], y=titanic['age'], kind='scatter')
# plt.show()


# # Group the data by gender ('sex') and survival status ('survived')
# grouped = titanic.groupby(['sex', 'survived'])['age'].describe()
#
# # Display the summary statistics
# print(grouped)