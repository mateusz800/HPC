import pandas as pd

df = pd.read_csv("minute_weather.csv").dropna()
df = df.sample(frac=1)
df.to_csv("minute_weather_shuffled.csv")