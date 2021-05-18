import pandas as pd
import matplotlib.pyplot as plt

df_result = pd.read_csv("result.csv")
df_data = pd.read_csv("Iris.csv")

y = pd.factorize(df_data["Species"])[0]

plt.subplot(1, 2, 1)
plt.title("Real data")
plt.scatter(df_data['SepalLengthCm'], df_data['SepalWidthCm'], c=y, cmap='gist_rainbow')
plt.xlabel('Sepa1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)

plt.subplot(1, 2, 2)
plt.title("Clustered using K-means")
plt.scatter(df_data['SepalLengthCm'], df_data['SepalWidthCm'], c=df_result["Cluster"], cmap='gist_rainbow')
plt.xlabel('Sepa1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)
plt.show()



