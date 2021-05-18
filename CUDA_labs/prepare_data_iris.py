import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Iris.csv").dropna()
df = df.drop(columns=["Id", "Species", "PetalLengthCm","PetalWidthCm"])

#shuffle
#df = df.sample(frac=1) 

# Scale the features
#X = MinMaxScaler().fit_transform(df)
#prepared_df = pd.DataFrame(X, columns = ['air_pressure','air_temp','avg_wind_speed','max_wind_speed','min_wind_speed','rain_accumulation','rain_duration','relative_humidity'])
df.to_csv("iris_prepared.csv")