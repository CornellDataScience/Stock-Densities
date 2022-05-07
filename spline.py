#%%
from functools import update_wrapper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, BSpline
from collections import defaultdict
from datetime import timedelta

# def get_knots(data):
#    '''
#    Currently evenly spaced knots, start and end at quantiles
#    Should not include max and min of data
#    '''
#    upper = np.quantile(data, 0.8)
#    lower = np.quantile(data, 0.2)
#    num_knots = int(2*len(data)**(1/5)) # suggested number of knots
#    return np.linspace(lower, upper, num_knots) 

def get_knots_fit(data):
   knots = get_knots(start, end)

def get_knots(all_ret):
   ''' NOT ROLLING WINDOW: uses all training data to determine placement of knots''' 
   lower = np.min(all_ret) / 1.1
   upper = np.max(all_ret) / 1.1
   return np.linspace(lower, upper, 5)

# def get_knots():
#    return np.linspace(-0.005, 0.005, 20)

def interior_knots(min, max, knots):
   ''' Returns interior knots only, and number of knots cut from both sides
   These should be interior knots as knots on the ends will be added automatically.'''

   # knots = get_knots()
   num_knots = len(knots)
   min_mask = knots > min
   left_zeros = num_knots - sum(min_mask)
   knots = knots[min_mask]
   max_mask = knots < max
   right_zeros = num_knots - sum(max_mask)
   knots = knots[max_mask]
   return knots, left_zeros, right_zeros


def get_ret_by_date(df):
   '''
   Dictionary using date as key, returns as value
   '''
   ret_by_date = defaultdict(list)
   for i in range(len(df)):
      ret_by_date[df["Dates"].iloc[i].date()].append(df["log_ret"].iloc[i])
   return ret_by_date

def bin_data(data):
   '''
   Return midpoints and masses
   '''
   hist, bin_edges = np.histogram(data, bins=50)
   # Get rid of 0s
   vary_edges = [bin_edges[0]]
   for i, e in enumerate(bin_edges[1:]):
      if hist[i] != 0: # keep edge as long as non zero
         vary_edges.append(e)
   # refit on new edges
   hist, bin_edges = np.histogram(data, bins=vary_edges)
   masses = hist / np.sum(hist)

   # print("masses", masses)
   midpoints = (bin_edges[:-1] + bin_edges[1:])/2

   return midpoints, masses

def plot_fit(midpoints, masses, spl, filename=None):
   # x2 = np.linspace(-0.002, 0.002, 200)
   x2 = midpoints
   y2 = splev(x2, spl)

   # plt.plot(midpoints, np.log(masses), 'o')
   # plt.plot(x2, y2, color="red")
   # plt.show()

   plt.plot(midpoints, masses, 'o')
   plt.plot(spl[0], np.exp(splev(spl[0], spl)), 'o', color="yellow")
   plt.plot(x2, np.exp(y2), color="red")
   if filename:
      plt.savefig(filename)
   plt.show()


#%%
if __name__ == "__main__":
   ret = pd.read_csv("data/SPY_minute.csv",skiprows=1)
   datetimes = pd.to_datetime(ret.Dates[1:], format='%m/%d/%y %H:%M')
   ret["Dates"][1:] = datetimes
   ret["Dates"][0] = ret["Dates"][1] - timedelta(minutes=1)
   ret["log_ret"] = np.log(ret["Close"]) - np.log(ret.shift(1)["Close"])
   ret = ret.dropna()
   ret_min = np.min(ret["log_ret"])
   ret_max = np.max(ret["log_ret"])

   knots = get_knots(ret["log_ret"])
   print("knots", knots)

   # Group data by date
   ret_by_date = get_ret_by_date(ret)

   pars = []
   # Get thetas
   for date, rets in ret_by_date.items():
      try:
         midpoints, masses = bin_data(rets)
         # pad two endpoints so that the annoying function would work
         midpoints = np.concatenate((np.array([ret_min]), midpoints, np.array([ret_max])))
         print("midpoints", midpoints)
         masses = np.concatenate((np.array([0]), masses, np.array([0])))
         # print("log_masses", log_masses)
         # for logspline
         # knots = get_knots(midpoints)
         # knots = get_knots_pred()
         # knots, left, right = interior_knots(np.min(midpoints), np.max(midpoints))
         print(np.min(knots) > np.min(midpoints), np.max(knots) < np.max(midpoints))
         spl = splrep(midpoints, np.log(masses), t=knots) 
         # spl = splrep(midpoints, np.log(masses))
         # print(spl[0])
         pars.append(spl[1]) # append thetas
         
         plot_fit(midpoints, masses, spl)
         
      except Exception as e:
         print(e)

   res = pd.DataFrame(pars)
   res.index = ret_by_date.keys()
   # res.to_csv("data/thetas.csv")

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
   res.to_csv("data/thetas_preds.csv")

   # %% Generating spline from theta predictions
   thetas = pd.read_csv("data/thetas_preds.csv", index_col=0)
   # thetas.index = pd.to_datetime(thetas.index)

   for i, date in enumerate(thetas.index):
      knots = get_knots_pred() # you don't know x. How to even get evenly spaced knots?
      spl = BSpline(knots, thetas.iloc[i], 3, extrapolate=True, axis=0)
      fig, ax = plt.subplots()
      xx = np.linspace(-0.01, 0.01, 50)
      # ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
      ax.plot(xx, np.exp(spl(xx)), 'b-', lw=4, alpha=0.7, label='BSpline')
      ax.grid(True)
      ax.legend(loc='best')
      plt.show()

# for date, rets in ret_by_date.items():



# %%
