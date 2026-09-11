from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = datasets.load_iris()
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)
#Question 1
#a
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
plt.hist(iris_df["sepal width (cm)"], bins=10, color='blue', alpha=0.7,edgecolor='black')
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()
#c
mean = np.mean(iris_df["sepal width (cm)"])
median = np.median(iris_df["sepal width (cm)"])
print(f"Mean: {mean}, Median: {median}")
#d
top27width = np.percentile(iris_df["sepal width (cm)"], [73])
print(f"73th Percentile: {top27width}")
#e
plt.scatter(iris_df["sepal length (cm)"], iris_df["sepal width (cm)"], color='red')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length vs Sepal Width")
plt.show()
plt.scatter(iris_df["sepal length (cm)"], iris_df["petal length (cm)"], color='green')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")
plt.show()
plt.scatter(iris_df["sepal length (cm)"], iris_df["petal width (cm)"], color='purple')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Sepal Length vs Petal Width")
plt.show()
plt.scatter(iris_df["sepal width (cm)"], iris_df["petal length (cm)"], color='orange')
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Width vs Petal Length")
plt.show()
plt.scatter(iris_df["sepal width (cm)"], iris_df["petal width (cm)"], color='cyan')
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Sepal Width vs Petal Width")
plt.show()
plt.scatter(iris_df["petal length (cm)"], iris_df["petal width (cm)"], color='magenta')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width")
plt.show()
#Question 2
#a
custom_bins = np.arange(3.3, 6.6, 0.3)
plt.hist(PlantGrowth["weight"], bins=custom_bins, color='red', alpha=0.7,edgecolor='black')
plt.xlabel("Weight")
plt.ylabel("Frequency")
plt.show()
plt.boxplot(PlantGrowth["weight"])
plt.xlabel("Weight")
plt.show()
#b
plant_df = pd.DataFrame(PlantGrowth)
sns.boxplot(x="group", y="weight", data=plant_df, palette="Set1")
plt.title("Distribution of Weight by Group")
plt.xlabel("Group")
plt.ylabel("Weight")
plt.show()
#d
trt1_weights = plant_df[plant_df["group"] == "trt1"]["weight"]
trt2_weights = plant_df[plant_df["group"] == "trt2"]["weight"]
min_trt2 = trt2_weights.min()
below_min = trt1_weights[trt1_weights < min_trt2]
percent_below_min = (len(below_min) / len(trt1_weights)) * 100
print(f"Percentage of trt1 weights below the minimum of trt2: {percent_below_min}%")
#e
filtered_df = PlantGrowth[PlantGrowth["weight"] > 5.5]
sns.barplot(data=filtered_df, x="group", y="weight", palette="deep")
plt.title("Average Weight per Group (Filtered > 5.5)")
plt.show()