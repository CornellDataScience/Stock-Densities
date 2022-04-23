
def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep
from collections import defaultdict
from datetime import timedelta

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
      
   print("masses", masses)
   midpoints = (bin_edges[:-1] + bin_edges[1:])/2
   spl = splrep(midpoints, np.log(masses)) #logspline
   x2 = np.linspace(-0.002, 0.002, 200)
   x2 = np.linspace(-0.002, 0.002, 200)
   y2 = splev(x2, spl)
   plt.plot(midpoints, np.log(masses), 'o')
   plt.plot(x2, y2, color="red")
   plt.show()
   # plt.plot(midpoints, masses, 'o')
   # plt.plot(x2, np.exp(y2), color="red")
   # plt.show()




# %%

# %%
