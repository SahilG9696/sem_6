import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from assingement_9 import titanic

df= pd.read_csv("titanic.csv")
print(df)
print(df.head())
print(df.tail())
print(df.describe())

# #A] Distribution plots
# # 1] Join plot
# # For Plot 1
# sns.jointplot(x = df['age'], y =df['fare'], kind ='scatter')
# plt.show()
# # For Plot 2
# sns.jointplot(x = df['age'], y =df['fare'], kind = 'hex')
# plt.show()
#
# #2] The Rug Plot
# sns.rugplot(df['fare'])
# plt.show()
#
# #B] Categorical Plots
# #1] The Bar Plot
# sns.barplot(x='sex', y='age', data=df)
# plt.show()
#
# #2] The count Plot
# sns.countplot(x='sex', data=df)
# plt.show()
#
# #3] The Box Plot
# sns.boxplot(x='sex', y='age',hue='survived', data=df)
# plt.show()
#
# #4] vivolinplot plot
# sns.violinplot(x='sex', y='age', data=df,hue='survived')
# plt.show()
#
# #C] Advanced Plots:
# # 1] The Strip Plot
# # a] False
# sns.stripplot(x='sex', y='age', data=df, jitter=False)
# plt.show()
# #b] True
# sns.stripplot(x='sex', y='age', data=df, jitter=True)
# plt.show()
#
# # 2] The Swarm Plot
# sns.swarmplot(x='sex', y='age', data=df, hue='survived')
# plt.show()


# D] Matrix plot
# 1] Heat Plot
titanic["sex"] = titanic["sex"].map({"male": 0, "female": 1})
titanic = titanic.select_dtypes(include=["number"])
corr = titanic.corr()
sns.heatmap(corr, annot=True)
plt.show()







