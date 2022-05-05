#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.collections import PolyCollection
import pandas as pd
import numpy as np
dists = pd.read_csv('data/pars_t_daily_spy.csv')



plt.rcParams["figure.figsize"] = (10,10)
fig = plt.figure()
ax = plt.axes(projection='3d')

deltas = pd.read_csv('data/arr.csv')
arr = deltas['deltas']
x = np.linspace(min(arr), max(arr), 1000)

def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]

#  verts = []
#  for i in dists.index:
    #  df = dists.loc[[i]].df
    #  mean = dists.loc[[i]].mu
    #  sd = dists.loc[[i]].sigma
    #  verts.append(polygon_under_graph(x, t.pdf(x, df, mean, sd)))
    #  ax.plot(x, t.pdf(x, df, mean, sd), i,
       #  'gray', lw=.5, zdir='y')
verts = [polygon_under_graph(x, t.pdf(x, d.df, d.mu, d.sigma)) for idx, d in dists.iterrows()]

facecolors = plt.colormaps['viridis_r'](np.linspace(0,1, len(verts)))

poly = PolyCollection(verts, facecolors=facecolors, alpha=0.7)
#  print("Poly: ", poly)
ax.add_collection3d(poly, zs=dists.index, zdir='y')
ax.set(xlim=(-0.001, 0.001), zlim=(0, 2500))

df = dists.loc[[dists.index[-1]]].df
mean = dists.loc[[dists.index[-1]]].mu
sd = dists.loc[[dists.index[-1]]].sigma

line, = ax.plot(x, t.pdf(x, df, mean, sd), dists.index[-1]+1,
       'red', lw=2, zdir='y', label='prediction')

ax.set_ylim(ax.get_ylim()[::-1])

ax.legend()
plt.savefig("myimage.eps", dpi=100000)
plt.show()
