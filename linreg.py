import numpy as np
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

    Returns:
        The mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    print("**Predicting MPG**")
    weights = np.array(load_data(DATA_PATH)[0])  # Load weights from the data file
    mpgs = np.array(load_data(DATA_PATH)[1])  # Load mpgs from the data file
    x = weights  # Take the weights for prediction
    x_shape = np.shape(x)
    print(f"Shape of x: {x_shape}\n")
    
    w = (9-21)/(4732-2648)
    b = 21 - w * 2648

    print(f"Weight: {w}, Bias: {b}\n")

    y_pred = predict(x, w, b)
    y_pred_shape = np.shape(y_pred)
    print(f"Shape of y_pred: {y_pred_shape}\n")
    print(f"Predicted MPG for weights {x} is {y_pred}")
    print(f"True MPG for weights {x} is {mpgs}")
    y_true = mpgs
    error = mse(y_true, y_pred)
    print(f"Mean Squared Error : {error}")
