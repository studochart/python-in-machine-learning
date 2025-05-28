# Seaborn Practice Questions - Solutions
# These are solutions to the practice questions from the Seaborn tutorial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size for all plots
plt.figure(figsize=(10, 6))

# Question 1: Create a simple line plot using Seaborn with random data
print("Solution to Question 1: Simple Line Plot")
# Generate random data
np.random.seed(42)
x = np.arange(0, 10, 0.1)
y = np.sin(x) + np.random.normal(0, 0.2, len(x))

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='x', y='y', data=df)
plt.title('Simple Line Plot with Random Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Question 2: Load the 'tips' dataset from Seaborn and create a scatter plot of 'total_bill' vs 'tip'
print("\nSolution to Question 2: Scatter Plot with Tips Dataset")
# Load the tips dataset
tips = sns.load_dataset('tips')

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.title('Scatter Plot of Total Bill vs Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.show()

# Question 3: Create a histogram of the 'tip' column from the tips dataset
print("\nSolution to Question 3: Histogram of Tips")
plt.figure(figsize=(10, 6))
sns.histplot(tips['tip'], kde=True, bins=20)
plt.title('Distribution of Tips')
plt.xlabel('Tip Amount ($)')
plt.ylabel('Frequency')
plt.show()

# Question 4: Create a box plot showing the distribution of 'tip' by 'day' from the tips dataset
print("\nSolution to Question 4: Box Plot of Tips by Day")
plt.figure(figsize=(10, 6))
sns.boxplot(x='day', y='tip', data=tips)
plt.title('Distribution of Tips by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Tip Amount ($)')
plt.show()

# Question 5: Create a violin plot showing the distribution of 'total_bill' by 'day' from the tips dataset
print("\nSolution to Question 5: Violin Plot of Total Bill by Day")
plt.figure(figsize=(10, 6))
sns.violinplot(x='day', y='total_bill', data=tips)
plt.title('Distribution of Total Bill by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill Amount ($)')
plt.show()

# Question 6: Create a count plot showing the frequency of each 'day' in the tips dataset
print("\nSolution to Question 6: Count Plot of Days")
plt.figure(figsize=(10, 6))
sns.countplot(x='day', data=tips)
plt.title('Frequency of Each Day in the Tips Dataset')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

# Question 7: Create a heatmap of the correlation matrix for the iris dataset
print("\nSolution to Question 7: Correlation Heatmap of Iris Dataset")
# Load the iris dataset
iris = sns.load_dataset('iris')

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(iris.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Iris Features')
plt.tight_layout()
plt.show()

# Question 8: Create a pair plot for the iris dataset with points colored by species
print("\nSolution to Question 8: Pair Plot of Iris Dataset")
sns.pairplot(iris, hue='species')
plt.suptitle('Pair Plot of Iris Dataset by Species', y=1.02)
plt.show()

# Question 9: Create a bar plot showing the average 'tip' by 'day' from the tips dataset
print("\nSolution to Question 9: Bar Plot of Average Tip by Day")
plt.figure(figsize=(10, 6))
sns.barplot(x='day', y='tip', data=tips, estimator=np.mean)
plt.title('Average Tip by Day')
plt.xlabel('Day of the Week')
plt.ylabel('Average Tip Amount ($)')
plt.show()

# Question 10: Change the theme of your plots to 'darkgrid' and create any plot of your choice
print("\nSolution to Question 10: Plot with Darkgrid Theme")
# Set the theme to darkgrid
sns.set_theme(style='darkgrid')

# Create a joint plot as an example
plt.figure(figsize=(10, 8))
sns.jointplot(x='total_bill', y='tip', data=tips, kind='reg')
plt.suptitle('Joint Plot of Total Bill vs Tip (Darkgrid Theme)', y=1.02)
plt.show()

# Bonus: Create a more advanced plot - FacetGrid
print("\nBonus: FacetGrid Example")
# Reset the theme to default
sns.set_theme()

# Create a FacetGrid
g = sns.FacetGrid(tips, col="time", row="sex", height=4)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
g.fig.suptitle('Tips by Total Bill, Time, and Gender', y=1.05)
plt.show()

# Bonus: Create a more advanced plot - Swarm Plot
print("\nBonus: Swarm Plot Example")
plt.figure(figsize=(12, 6))
sns.swarmplot(x='day', y='total_bill', hue='sex', data=tips)
plt.title('Swarm Plot of Total Bill by Day and Gender')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill Amount ($)')
plt.show()

print("\nAll solutions have been demonstrated!")



# Seaborn Quiz Answers

Here are the correct answers to the Seaborn quiz from the tutorial:

1. **What is Seaborn?**
   - ✅ A statistical data visualization library based on Matplotlib
   - ❌ A low-level plotting library in Python
   - ❌ A web framework for creating interactive dashboards
   - ❌ A machine learning library for predictive modeling

2. **Which function would you use to create a scatter plot in Seaborn?**
   - ❌ sns.plot()
   - ✅ sns.scatterplot()
   - ❌ sns.scatter()
   - ❌ sns.pointplot()

3. **What does the 'hue' parameter do in Seaborn plots?**
   - ❌ Changes the overall color of the plot
   - ❌ Adjusts the brightness of the plot
   - ✅ Maps a categorical variable to different colors
   - ❌ Controls the transparency of plot elements

4. **Which Seaborn plot is best for visualizing the distribution of a single variable?**
   - ❌ Scatter plot
   - ❌ Heatmap
   - ✅ Histogram or KDE plot
   - ❌ Line plot

5. **What is a violin plot in Seaborn?**
   - ❌ A plot showing the relationship between two categorical variables
   - ✅ A plot that combines a box plot with a kernel density estimate
   - ❌ A plot showing the correlation between multiple variables
   - ❌ A plot specifically designed for time series data

6. **Which Seaborn function would you use to create a correlation matrix heatmap?**
   - ✅ sns.heatmap()
   - ❌ sns.corrplot()
   - ❌ sns.matrixplot()
   - ❌ sns.correlationmap()

7. **What is a pair plot in Seaborn?**
   - ❌ A plot showing only pairs of variables that are highly correlated
   - ✅ A grid of plots showing relationships between pairs of variables
   - ❌ A plot showing only two variables at a time
   - ❌ A specialized plot for comparing exactly two datasets

8. **How do you change the style of Seaborn plots?**
   - ❌ sns.style('darkgrid')
   - ✅ sns.set_theme(style='darkgrid')
   - ❌ sns.set_style = 'darkgrid'
   - ❌ sns.theme('darkgrid')

9. **Which of these is NOT a built-in dataset in Seaborn?**
   - ❌ iris
   - ❌ tips
   - ❌ titanic
   - ✅ boston

10. **What's the relationship between Matplotlib and Seaborn?**
    - ❌ They are completely independent libraries with no connection
    - ❌ Matplotlib is built on top of Seaborn
    - ✅ Seaborn is built on top of Matplotlib
    - ❌ They are competing libraries that cannot be used together

## Explanation of Answers

1. Seaborn is a statistical data visualization library that builds on top of Matplotlib to provide a higher-level interface for creating attractive and informative statistical graphics.

2. The correct function for creating scatter plots in Seaborn is `sns.scatterplot()`. This function creates scatter plots with the ability to show relationships between up to five variables.

3. The `hue` parameter in Seaborn plots maps a categorical variable to different colors, allowing you to visualize an additional dimension of your data.

4. Histograms and KDE (Kernel Density Estimate) plots are best for visualizing the distribution of a single variable. In Seaborn, you can create these using `sns.histplot()` or `sns.kdeplot()`.

5. A violin plot combines a box plot with a kernel density estimate, showing both the distribution of the data and its probability density.

6. `sns.heatmap()` is used to create heatmaps, which are commonly used to visualize correlation matrices.

7. A pair plot creates a grid of scatter plots for all pairs of variables in a dataset, with histograms or KDE plots on the diagonal.

8. The correct way to change the style of Seaborn plots is using `sns.set_theme(style='darkgrid')`.

9. While 'iris', 'tips', and 'titanic' are built-in datasets in Seaborn, 'boston' is not. The Boston housing dataset is commonly found in scikit-learn, not Seaborn.

10. Seaborn is built on top of Matplotlib, extending its functionality with statistical visualizations and providing a higher-level interface.

