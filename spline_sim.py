#%%
import spline
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep

# %%
data = dict()
with open('simulated_data.csv', newline='') as csvfile:
    sim_reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    for i, row in enumerate(sim_reader):
        data[i] = row
# %%
pars = []

knots = spline.get_knots(np.array(list(data.values())))
knots_all = np.linspace(-30, 30, 30)
for i, batch in data.items():
    try:
        midpoints, masses = spline.bin_data(batch)
        knots, left, right = spline.interior_knots(np.min(midpoints), np.max(midpoints), knots_all)
        spl = splrep(midpoints, np.log(masses), t=knots)
        thetas = spl[1]
        # thetas = np.concatenate((np.zeros(left), thetas, np.zeros(right))) # pad zeros to excluded knots
        pars.append(thetas) # append thetas
        if i % 10 == 0:
            print("spl", spl[0])
            # print("knot", knots)
            print("par", pars[i])
            print("knots", len(spl[0]))
            print(len(pars[i]))
            # spline.plot_fit(midpoints, masses, spl)
    except Exception as e:
        print(e)
        print(i)

# %%
