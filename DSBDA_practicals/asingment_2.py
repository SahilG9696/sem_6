import pandas as pd
import numpy as np

# Displaying the dataset okk
df = pd.read_csv("StudentsPerformance.csv")
print(df)

# calculating missing values count for each colume in the dataset okk
print(df.isnull().sum())

# Scan all variables for missing values and inconsistencies. If there are missing values and/or
# inconsistencies, use any of the suitable techniques to deal with them.


# showing the True and False for Nan And Other data
print(df.notnull())

# replacing the Nan with 0 in the dataset inplace=True used to print original dataset with modification without creating the other copy
df.fillna(0,inplace=True)
print(df)

# Filling NaN using forward fill (pad method) i.e ffill can replace the Nan with forward value
df.ffill(inplace=True)
print(df)

# Filling NaN using forward fill (pad method) i.e ffill can replace the Nan with backward value
df.bfill(inplace=True)
print(df)

# Filling NaN using fillna for a perticular colume and make all above the comment okkk
print(df["Math_Score"].fillna(45))

# Replacing the NaN values with 85 okkk
print(df.replace(to_replace = np.nan, value = 85))

# using dropna() function
print(df.dropna())

# using dropna() function
print(df.dropna(how = 'all',inplace=True))

# using dropna() function
print(df.dropna(axis = 1))  #df.dropna(axis=1) will remove all columns that contain at least one missing value (NaN)

# making new data frame with dropped NA values
df2 = df.dropna(axis = 0, how ='any')
print(df2)

                                        # OUTLAYERS
# Scan all numeric variables for outliers. If there are outliers, use any of the suitable
# techniques to deal with them.


#1 BOX PLOT
import seaborn as sns   #Seaborn is a library built on top of matplotlib which used to plots and charts.
import matplotlib.pyplot as plt   #This line tells Python to bring in the matplotlib library and use the pyplot module, which is typically shortened as plt.

# Create a box plot for Math Scores
sns.boxplot(x=df['Math_Score'])     #The x= part tells Seaborn to plot math scores along the x-axis.
                                    #sns.boxplot() is a function from Seaborn, a Python library for data visualization.
# Show the plot
plt.title('Box plot of the Math_Score attribute ')
plt.show()  #When you create a graph using matplotlib, it doesn't automatically appear. You need to use plt.show() to tell Python to display the graph you’ve made. It's like telling the program, "Hey, I’m ready to see my plot now!"
# Position of the Outlier
print(np.where(df['Math_Score']>70.0))
df.boxplot()
plt.show()

#2 SCATTERPLOT
fig, ax = plt.subplots(figsize=(18, 10))
ax.scatter(df['Club_Join_Date'], df['Reading_Score'])

# x-axis label
ax.set_xlabel('Club_join_Date')

# y-axis label
ax.set_ylabel('Reading Score')
plt.show()

# Position of the Outlier
print(np.where((df['Reading_Score']>75.0) & (df['Club_Join_Date']==2020)))

#3 Z SCORE
from scipy import stats
# NOTE:- if in the colume value is NaN then first replace that then apply this
#The zscore() function calculates the z-score of the data in the Placement_Offer_Count column of the DataFrame (df).
# A z-score tells you how many standard deviations a particular value is from the mean of the data.
z = np.abs(stats.zscore(df['Placement_Offer_Count']))
print(z)

#SEE THIS AGAIN FOR CORRECTION
# #4 IQR
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Math_Score'] < lower_bound) | (df['Math_Score'] > upper_bound)]
print(outliers)

# You can remove outliers if desired
df = df[(df['Math_Score'] >= lower_bound) & (df['Math_Score'] <= upper_bound)]

#Removing OUTLAYERS
print("Old Shape: ", df.shape)
df.drop(upper_bound[0], inplace = True)
df.drop(lower_bound[0], inplace = True)

print("New Shape: ", df.shape)

                                #DATA TRANSFORMATION
#Apply data transformations on at least one of the variables. The purpose of this
# #transformation should be one of the following reasons: to change the scale for better
# understanding of the variable, to convert a non-linear relation into a linear one, or to
# decrease the skewness and convert the distribution into a normal distribution.


#1 LOG TRANSFORMATION OKKKK
from sklearn.preprocessing import MinMaxScaler

# MinMax Scaling
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[['Math_Score', 'Reading_Score', 'Placement_Offer_Count']])

# Add the normalized values back to the DataFrame
df[['Normalized_Math_Score', 'Normalized_Reading_Score', 'Normalized_Placement_Offer_Count']] = normalized_data

# Plot the original and normalized data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df[['Math_Score', 'Reading_Score', 'Placement_Offer_Count']].values.flatten(), bins=20, color='blue', alpha=0.7)
plt.title('Before Normalization')

plt.subplot(1, 2, 2)
plt.hist(df[['Normalized_Math_Score', 'Normalized_Reading_Score', 'Normalized_Placement_Offer_Count']].values.flatten(), bins=20, color='green', alpha=0.7)
plt.title('After Normalization')

plt.tight_layout()
plt.show()


#2 Scaling/Normalization
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
print(scaler.fit_transform(df[['Math_Score', 'Reading_Score', 'Placement_Offer_Count']]))

#3 Power Transformation
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
print(pt.fit_transform(df[['Math_Score', 'Reading_Score', 'Placement_Offer_Count']]))
