#%%
from functools import update_wrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from collections import defaultdict
from datetime import timedelta

def get_knots(data):
   '''
   Currently evenly spaced knots, start and end at quantiles
   Should not include max and min of data
   '''
   upper = np.quantile(data, 0.8)
   lower = np.quantile(data, 0.2)
   num_knots = int(2*len(data)**(1/5)) # suggested number of knots
   return np.linspace(lower, upper, num_knots) 


#%%
df = pd.read_csv("data/SPY_minute.csv",skiprows=1)
datetimes = pd.to_datetime(df.Dates[1:], format='%m/%d/%y %H:%M')
df["Dates"][1:] = datetimes
df["Dates"][0] = df["Dates"][1] - timedelta(minutes=1)
df["log_ret"] = np.log(df["Close"]) - np.log(df.shift(1)["Close"])
df = df.dropna()

# Group data by date

ret_by_date = defaultdict(list)
for i in range(len(df)):
    ret_by_date[df["Dates"].iloc[i].date()].append(df["log_ret"].iloc[i])

pars = []
# Get thetas
for date, rets in ret_by_date.items():
   hist, bin_edges = np.histogram(rets, bins=50)
   # Get rid of 0s
   vary_edges = [bin_edges[0]]
   for i, e in enumerate(bin_edges[1:]):
      if hist[i] != 0: # keep edge as long as non zero
         vary_edges.append(e)
   # refit on new edges
   hist, bin_edges = np.histogram(rets, bins=vary_edges)
   masses = hist / np.sum(hist)

   # print("masses", masses)
   midpoints = (bin_edges[:-1] + bin_edges[1:])/2
   # print("log_masses", log_masses)
    # for logspline
   spl = splrep(midpoints, np.log(masses), t=get_knots(midpoints)) 
   # spl = splrep(midpoints, np.log(masses))
   print(spl[0])
   pars.append(spl[1]) # append thetas
   
   '''
   # x2 = np.linspace(-0.002, 0.002, 200)
   x2 = midpoints
   y2 = splev(x2, spl)

   # plt.plot(midpoints, np.log(masses), 'o')
   # plt.plot(x2, y2, color="red")
   # plt.show()

   plt.plot(midpoints, masses, 'o')
   plt.plot(spl[0], np.exp(splev(spl[0], spl)), 'o', color="yellow")
   plt.plot(x2, np.exp(y2), color="red")
   plt.show()
   '''

res = pd.DataFrame(pars)
res.index = ret_by_date.keys()
res.to_csv("data/thetas.csv")

# %% Time series on coefficients
import arima

df = pd.read_csv("data/thetas.csv", index_col=0)
print(df.head())
df = df.fillna(0)

# idx = int(len(df)*0.8)
# train, test = df.iloc[:idx], df.iloc[idx:]
# train.head(), test.head()

pred = pd.DataFrame(index=df.index)
for par in df.columns:
   arima.adf_check(df[par]) # Stationary -- no differencing needed
   print(f"=================== {par} ===================")
   results = arima.arma(1, 10, 1, 10, pd.DataFrame(df[par]), par, "BIC")
   results["data"][[par,'Predicted_Values']].plot()
   pred[par] = results["data"][['Predicted_Values']]
   plt.title(par)
   plt.show()

# Store results
print(pred)
res = pred.dropna()
# res.index = ret_by_date.keys() #TODO: fix index
res.to_csv("data/thetas_preds.csv")

# %% Generating spline from theta predictions

