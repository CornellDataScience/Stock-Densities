import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import csv

def simulate_data(
        loc_fn1=lambda x: -x/25, 
        loc_fn2=lambda x: x/25, 
        var_fn1=lambda x: 1,
        var_fn2=lambda x: 1,
        n=1000, t=100, show=True, name=""):
    samples = dict()
    for i in range(t):
        samples[i] = np.append(
            normal(loc=loc_fn1(i), scale=var_fn1(i), size=n), 
            normal(loc=loc_fn2(i), scale=var_fn2(i), size=n)
        )

    with open(name+"simulated_data.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(t):
            writer.writerow(samples[i])

    if show:
        for i in range(0,t,10):
            plt.hist(samples[i], bins=20)
            plt.show()

if __name__ == "__main__":
    simulate_data(loc_fn1=lambda x: -x/10)
