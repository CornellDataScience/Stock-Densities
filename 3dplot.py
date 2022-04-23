#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
dists = pd.read_csv('data/pars_t_daily_spy.csv')



plt.rcParams["figure.figsize"] = (10,10)
fig = plt.figure()
ax = plt.axes(projection='3d')

deltas = pd.read_csv('data/arr.csv')
arr = deltas['deltas']
x = np.linspace(min(arr), max(arr), 1000)

for i in dists.index:
    df = dists.loc[[i]].df
    mean = dists.loc[[i]].mu
    sd = dists.loc[[i]].sigma
    ax.plot(x, t.pdf(x, df, mean, sd), i,
       'gray', lw=.5, zdir='y')
    
line, = ax.plot(x, t.pdf(x, df, mean, sd), dists.index[-1]+1,
       'red', lw=1, zdir='y', label='prediction')

ax.set_ylim(ax.get_ylim()[::-1])

ax.legend()
plt.savefig("myimage.eps", dpi=100000)
plt.show()
