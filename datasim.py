import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
import csv

def sim_densities_data(
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
    print(samples[1])

    with open(name+"simulated_data.csv", 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(t):
            writer.writerow(samples[i])

    if show:
        for i in range(0,t,10):
            plt.hist(samples[i], bins=20)
            plt.show()


def sim_stock_data(
    mu = 0.001,
    sigma = 0.01,
    start_price = 5):
    returns = np.random.normal(loc=mu, scale=sigma, size=100)
    price = start_price*(1+returns).cumprod()
    return price

if __name__ == "__main__":
    sim_densities_data(loc_fn1=lambda x: -x/10)
    plt.plot(sim_stock_data())
    plt.show()

