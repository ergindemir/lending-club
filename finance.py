import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_sp_data():
    df = pd.read_csv('data/sp500.csv')
    df.Date = pd.to_datetime(df.Date)
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def volatility(df,ndays = 21):    
    ret = np.diff(df.Close)/df.Close[:-1]
    vol = [np.std(ret[n-ndays:n]) * 15.9  for n in range(ndays,len(ret)+1)]
    df_vol = df[['Date']]
    df_vol['Volatility'] = vol[0]
    df_vol.Volatility[ndays:] = vol
    df_vol.Date = df_vol.Date.values.astype('datetime64[M]') 
    df_vol = df_vol.groupby('Date').mean()
    df_vol['Date'] = df_vol.index
    return df_vol
      
df_vol = volatility(get_sp_data(),42) 
plt.plot(df_vol.Volatility)

