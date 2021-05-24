import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("minute_weather.csv").dropna()
df = df.drop(columns=["id","rowID","hpwren_timestamp",'air_pressure','air_temp','rain_accumulation','rain_duration','relative_humidity'])
#shuffle'
#df = df.sample(frac=1) 

# Scale the features (to range 0-1)
X = MinMaxScaler().fit_transform(df)
# id,rowID,hpwren_timestamp,air_pressure,air_temp,avg_wind_direction,avg_wind_speed,max_wind_direction,max_wind_speed,min_wind_direction,min_wind_speed,rain_accumulation,rain_duration,relative_humidity

prepared_df = pd.DataFrame(X, columns = ['avg_wind_direction','avg_wind_speed','max_wind_direction','max_wind_speed','min_wind_direction','min_wind_speed'])
prepared_df.to_csv("data2.csv")