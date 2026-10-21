import numpy as np
from pathlib import Path
WEIGHT_COL = 4
MPG_COL = 0
DATA_PATH = Path(__file__).resolve().parent / "data" / "auto-mpg.data"


def load_data(path):
    """Load vehicle weight and fuel economy from the UCI Auto MPG file.

    Args:
        path: Location of the whitespace-delimited auto-mpg.data file.

    Returns:
        A tuple (weights, mpgs) of 1-D float arrays, both of length N,
        where N is the number of cars. Weights are in pounds and mpgs
        are in miles per gallon.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If a loaded column has a token that is not a number
            (raised by np.loadtxt), or a value that is NaN or infinite
            (raised by this function).
    """ ##Thank you Claude

    x, y = np.loadtxt(fname=path, usecols= (WEIGHT_COL, MPG_COL), unpack = True)
    x_isfinite = np.isfinite(x)
    y_isfinite = np.isfinite(y)

    if not (x_isfinite.all() and y_isfinite.all()):
        total_size = x.size + y.size
        invalid_count = total_size - (x_isfinite.sum() + y_isfinite.sum())

        raise ValueError(f"Invalid (non finite, NaN, infinity) data in {path}. {invalid_count} Invalid values out of {total_size}.")
        
    return x, y

    
if __name__ == "__main__":
    print("**Loading Data**")
    print(f"Data Path: {DATA_PATH}\nExists: {DATA_PATH.exists()}\n")
    x, y = load_data(DATA_PATH)
    print(f"Data loaded successfully.")
    print (f"Minimum Weight: {x.min()}\nMaximum Weight: {x.max()}\nMinimum MPG: {y.min()}\nMaximum MPG: {y.max()}\n")
    print(f"Shape of x: {np.shape(x)}\nShape of y: {np.shape(y)}\n")