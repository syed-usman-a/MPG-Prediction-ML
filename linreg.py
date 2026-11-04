import matplotlib.pyplot as plt
import numpy as np

from load_data import DATA_PATH, load_data, load_multiple_feature_data


def predict(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """Return one prediction per input in the 1-D array x."""
    return w * x + b


def predict_multiple_features(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict from x shaped (cars, features) and w shaped (features,)."""
    return x @ w + b


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean squared error; raise ValueError for mismatched shapes."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    return np.mean((y_true - y_pred) ** 2)


def training(x: np.ndarray, y_true: np.ndarray, w: float, b: float, learning_rate: float, iterations: int) -> tuple[float, float, list[float]]:
    """Train on 1-D x and y_true; return slope, bias, and MSE history.

    Scale large input values before training. Each history entry records
    MSE after that iteration's parameter update.
    """
    mse_history = []
    for _ in range(iterations):
        predicted_y = predict(x, w, b)
        residuals = y_true - predicted_y
        gradient_bias = -2 * np.mean(residuals)
        gradient_weight = -2 * np.mean(x * residuals)

        b = b - learning_rate * gradient_bias
        w = w - learning_rate * gradient_weight
        mse_history.append(mse(y_true, predict(x, w, b)))

    return w, b, mse_history

def training_multiple_features(x: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: float, learning_rate: float, iterations: int) -> tuple[np.ndarray, float, list[float]]:
    """Train on 2-D x and y_true; return slope, bias, and MSE history.

    Scale large input values before training. Each history entry records
    MSE after that iteration's parameter update.
    """

    mse_history = []
    for _ in range(iterations):
        prediction = predict_multiple_features(x, w,b)
        residuals = y_true - prediction

        gradients_coefficients = (-2/len(y_true)) * (x.T @ residuals)
        gradient_bias = -2 * np.mean(residuals)

        w = w - learning_rate * gradients_coefficients
        b = b - learning_rate * gradient_bias
        
        mse_history.append(mse(y_true, predict_multiple_features(x, w, b)))

    return w, b, mse_history


def fit_closed_form(x: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Return the least-squares slope and bias for nonconstant 1-D x."""
    x_mean = np.mean(x)
    y_mean = np.mean(y_true)
    x_centered = x - x_mean
    y_centered = y_true - y_mean
    w = np.sum(x_centered * y_centered) / np.sum(x_centered ** 2)
    b = y_mean - w * x_mean
    return w, b


def plot_predictions(weights: np.ndarray, mpgs: np.ndarray, starting_predictions: np.ndarray, trained_predictions: np.ndarray) -> None:
    """Plot aligned 1-D predictions against vehicle weights in pounds."""
    plt.figure()
    plt.scatter(weights, mpgs, color="orange", label="Actual MPG")
    order = np.argsort(weights)
    plt.plot(
        weights[order], starting_predictions[order], color="red", label="Starting model"
    )
    plt.plot(
        weights[order], trained_predictions[order], color="blue", label="Trained model"
    )
    plt.xlabel("Weight (lb)")
    plt.ylabel("Fuel economy (mpg)")
    plt.legend()


def plot_training_loss(mse_history: list[float]) -> None:
    """Plot MSE in MPG squared after each training iteration."""
    plt.figure()
    plt.plot(range(1, len(mse_history) + 1), mse_history)
    plt.xlabel("Iterations")
    plt.ylabel("MSE (MPG^2)")


def run_single_feature_example(weights: np.ndarray, mpgs: np.ndarray) -> None:
    """Compare the starting model, mean baseline, training, and closed form.

    Inputs are aligned 1-D arrays of weight in pounds and fuel economy in MPG.
    """
    print("\nSingle feature example: weight -> MPG")
    x_scaled = weights / 1000

    # Preserve the manually chosen starting line, converting its slope to
    # MPG per thousand pounds for training.
    w = (9 - 27) / (4732 - 2130)
    b = 21 - w * 2648
    w_scaled = w * 1000
    starting_predictions = predict(x_scaled, w_scaled, b)
    print(f"Starting slope (MPG/1,000 lb): {w_scaled}, bias: {b}")
    print(f"Starting MSE: {mse(mpgs, starting_predictions)}")

    mean_predictions = np.full_like(mpgs, np.mean(mpgs))
    print(f"Mean baseline MSE: {mse(mpgs, mean_predictions)}")

    trained_w, trained_b, mse_history = training(x_scaled, mpgs, w_scaled, b, learning_rate=0.01, iterations=10000)
    trained_predictions = predict(x_scaled, trained_w, trained_b)
    print(f"Trained slope: {trained_w}, bias: {trained_b}")
    print(f"Trained MSE: {mse(mpgs, trained_predictions)}")

    closed_w, closed_b = fit_closed_form(x_scaled, mpgs)
    closed_predictions = predict(x_scaled, closed_w, closed_b)
    print(f"Closed-form slope: {closed_w}, bias: {closed_b}")
    print(f"Closed-form MSE: {mse(mpgs, closed_predictions)}")

    plot_predictions(weights, mpgs, starting_predictions, trained_predictions)
    plot_training_loss(mse_history)


def run_multiple_feature_example(features : np.ndarray, mpgs : np.ndarray) -> None:
    """Practise predictions and gradients on two made-up cars.

    Columns contain weight in thousands of pounds and horsepower.
    Coefficients are chosen manually; training updates are not implemented yet.
    """
    print("\nMultiple-feature example: weight + horsepower -> MPG")

    features_scaled = features / np.array([1000, 100])

    weight = np.zeros(features_scaled.shape[1])
    bias = np.mean(mpgs)

    weight, bias, mse_history = training_multiple_features(features_scaled, mpgs, weight, bias, 0.01, 10000)

    predictions = predict_multiple_features(features_scaled, weight, bias)
    ##print(f"Predictions: {predictions}")
    print(f"MSE: {mse_history[-1]}")
    plot_training_loss(mse_history)

    design_matrix = np.column_stack((features_scaled, np.ones(len(mpgs))))

    print(np.shape(design_matrix))

    parameters, residual_sums, rank, singular_values = np.linalg.lstsq(design_matrix, mpgs, rcond=None)
    w_closed = parameters[:2]
    b_closed = parameters[2]

    predictions_closed = predict_multiple_features(features_scaled, w_closed, b_closed)
    print(mse(mpgs, predictions_closed))

def split_data(x, y, test_fraction = 0.2, seed = 42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))

    test_size = int(len(y) * test_fraction)
    test_indices = indices[:test_size]
    train_indices = indices[test_size: ]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def main() -> None:
    """Load the dataset once and run the two learning examples."""
    weights, mpgs_single = load_data(DATA_PATH)
    features, mpgs = load_multiple_feature_data(DATA_PATH)
    
    run_single_feature_example(weights, mpgs_single)
    run_multiple_feature_example(features, mpgs)

    x_train, x_test, y_train, y_test = split_data(features, mpgs)
    print(np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test))

    # Uncomment to display the figures created by the single-feature example.
    #plt.show()


if __name__ == "__main__":
    main()
