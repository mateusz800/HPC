import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("minute_weather.csv").dropna()
df = df.drop(columns=["id","rowID","hpwren_timestamp","avg_wind_direction","max_wind_direction","min_wind_direction"])
#shuffle
#df = df.sample(frac=1) 

# Scale the features
X = MinMaxScaler().fit_transform(df)
prepared_df = pd.DataFrame(X, columns = ['air_pressure','air_temp','avg_wind_speed','max_wind_speed','min_wind_speed','rain_accumulation','rain_duration','relative_humidity'])
prepared_df.to_csv("minute_weather_prepared.csv")