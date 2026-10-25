import numpy as np
import matplotlib.pyplot as plt
from load_data import load_data, DATA_PATH

def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Predict the target values using a linear model.

    Args:
        x: 1-D array of input values.
        w: The slope of the linear model.
        b: The intercept of the linear model.

    Returns:
        The predicted target values.
    """
    return (w * x + b)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the mean squared error between true and predicted values.

    Args:
        y_true: 1-D array of true values.
        y_pred: 1-D array of predicted values.

    Raises:
        ValueError: If the shapes of y_true and y_pred do not match.
        
    Returns:
        The mean squared error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    return np.mean((y_true - y_pred) ** 2)

def training (x: np.ndarray, y_true: np.ndarray, w: float, b: float, learning_rate: float, iterations: int) -> tuple[float, float, list[float]]:

    mse_list = []
    for step in range(iterations):
        
        predicted_y = predict(x, w, b)
        residuals = y_true - predicted_y
        partial_b = -2 * np.mean(residuals)
        partial_w = -2 * np.mean(x * residuals)

        b = b - learning_rate * partial_b
        w = w - learning_rate * partial_w

        mse_list.append(mse(y_true, predict(x, w, b)))

    return w, b, mse_list


if __name__ == "__main__":
    print("**Predicting MPG**\n")


    #load data
    weights = np.array(load_data(DATA_PATH)[0])
    mpgs = np.array(load_data(DATA_PATH)[1])
    x = weights
    x_scaled = x / 1000


    #starting parameters
    w = (9-27)/(4732-2130)
    w_scaled = w * 1000
    b = 21 - w * 2648
    print(f"Starting Slope: {w}, Intercept: {b}")


    #starting mse
    y_pred = predict(x_scaled, w_scaled, b)
    y_true = mpgs
    error = mse(y_true, y_pred)
    print(f"Starting Mean Squared Error : {error}")

    mpg_mean = np.mean(mpgs)
    print(f"Mean MPG: {mpg_mean}")
    y_mean = np.full_like(mpgs, mpg_mean)
    error_mean = mse(y_true, y_mean)
    print(f"Mean Squared Error (using mean): {error_mean}")

    #training
    iterations = 10000
    learning_rate = 0.01
    w_new, b_new, mse_array = training(x_scaled, y_true, w_scaled, b, learning_rate, iterations)
    y_new = predict(x_scaled, w_new, b_new)
    print(f"New Mean Squared Error : {mse(y_true, y_new)}\n")


    x_mean = np.mean(x_scaled)
    x_centered = (x_scaled - x_mean)
    y_centered = (y_true - mpg_mean)

    w_closed = np.sum((x_centered * y_centered))/(np.sum(x_centered**2))
    b_closed = mpg_mean - (w_closed * x_mean)

    print(f"Closed Form Weight : {w_closed}")
    print(f"Closed Form Bias : {b_closed}")
    y_closed = predict(x_scaled, w_closed, b_closed)
    print(f"Closed Form MSE : {mse(y_true, y_closed)}")

    
    #plotting
    plt.scatter(x, y_true, color = "orange")
    plt.xlabel("Weight (lb)")
    plt.ylabel("Fuel economy (mpg)")

    order = np.argsort(x)
    plt.plot(x[order], y_pred[order], color = "red")
    plt.plot(x[order], y_new[order], color = "blue")
    
    plt.figure()
    plt.plot(range(1, iterations+1), mse_array)
    plt.xlabel("Iterations")
    plt.ylabel("MSE (MPG^2)")

    plt.show()