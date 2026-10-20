import numpy as np


def load_data(path):
    data = np.loadtxt(fname="data/auto-mpg.data", usecols= (0, 4))

    x = data[:, 1] #weights
    print(x.shape)

    y = data[:, 0] #mpgs
    print(y.shape)

    return x, y

weight, mpg = load_data("data/auto.mpg-data")

