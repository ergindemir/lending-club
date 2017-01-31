import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('data/sp500.csv')
df.Date = pd.to_datetime(df.Date)
df = df.sort_values(by='Date').reset_index(drop=True)
ret = np.diff(df.Close)/df.Close[:-1]
vol = [np.std(ret[n-21:n]) for n in range(21,len(ret)+1)]

df['Volatility'] = vol[0]
df.Volatility[21:] = vol
plt.plot(df.Date,df.Volatility)

