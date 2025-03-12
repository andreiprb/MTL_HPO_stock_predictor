import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import yfinance as yf

from sklearn.metrics import mean_absolute_error, r2_score


def min_max_scale(data):
    """
    Perform min-max scaling on the input data.

    Args:
        data (array): Input data to be normalized.

    Returns:
        tuple: (normalized_data, min_val, max_val) where:
            normalized_data: Data scaled to range [0,1]
            min_val: Minimum value from original data
            max_val: Maximum value from original data
    """
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data, min_val, max_val


def inverse_scale(data, min_val, max_val):
    """
    Inverse transform min-max scaled data back to original scale.

    Args:
        data (array): Normalized data to be transformed back.
        min_val (float): Minimum value from original data.
        max_val (float): Maximum value from original data.

    Returns:
        array: Data transformed back to original scale.
    """
    return data * (max_val - min_val) + min_val


def create_dataset(dataset, look_back):
    """
    Create a time series dataset with sliding window approach.

    Args:
        dataset (array): Input time series data.
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple: (X, Y) where:
            X: Input sequences of shape (samples, look_back)
            Y: Target values corresponding to the next value after each sequence
    """
    X, Y = [], []

    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])

    return np.array(X), np.array(Y)


def split_train_test(X, y, test_ratio):
    """
    Split data into training and test sets.

    Args:
        X (array): Input features.
        y (array): Target values.
        test_ratio (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) split data arrays.
    """
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test


def prepare_stock_data(ticker, start_date, end_date, look_back, test_ratio=0.05, verbose=False):
    """
    Centralized function to prepare stock data for model training.

    This function:
    1. Downloads stock data for the specified ticker
    2. Normalizes the data
    3. Creates time series sequences
    4. Splits into training and test sets

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data download (YYYY-MM-DD)
        end_date (str): End date for data download (YYYY-MM-DD)
        look_back (int): Number of previous time steps to use as input features
        test_ratio (float, optional): Proportion of data to use for testing. Defaults to 0.05.
        verbose (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - data_splits: (X_train, y_train, X_test, y_test)
            - scaling_info: (min_val, max_val)
            - dates: Original data dates
            - original_data: Original stock price data
    """
    if verbose:
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")

    data = yf.download(ticker, start=start_date, end=end_date, progress=verbose)

    if data.empty or len(data) < look_back + 10:
        if verbose:
            print(f"Insufficient data for {ticker}")
        return None

    close_data = data[['Close']].dropna()
    original_data = close_data.values
    normalized_data, min_val, max_val = min_max_scale(original_data)

    X, y = create_dataset(normalized_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_ratio)

    if verbose:
        print(
            f"Data prepared: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    return {
        'data_splits': (X_train, y_train, X_test, y_test),
        'scaling_info': (min_val, max_val),
        'dates': data.index,
        'original_data': original_data
    }


def plot_results(ticker, results, save=False, transfer_learning=False):
    """
    Plot the results of model training and evaluation with a continuous real data line.
    For transfer learning mode, predicted data is shown as a single connected purple line.

    Args:
        ticker (str): Stock ticker symbol.
        results (dict): Dictionary containing model predictions, dates, and metrics.
            Expected format:
            {
                'predictions': (train_preds_original, test_preds_original, y_train_original, y_test_original),
                'dates': (train_dates, test_dates),
                'metrics': {'original_rmse': float, 'normalized_rmse': float}
            }
        save (bool, optional): Whether to save the plot to a file. Defaults to False.
        transfer_learning (bool, optional): Whether the plot is for a transfer learning model.
                                           Defaults to False.

    Returns:
        None: Displays a plot showing real vs. predicted values for training and test data.
    """
    train_preds_original, test_preds_original, y_train_original, y_test_original = results['predictions']
    train_dates, test_dates = results['dates']
    metrics = results['metrics']

    min_train_len = min(len(train_dates), len(y_train_original))
    min_test_len = min(len(test_dates), len(y_test_original))

    train_dates = train_dates[:min_train_len]
    y_train_original = y_train_original[:min_train_len]
    train_preds_original = train_preds_original[:min_train_len]

    test_dates = test_dates[:min_test_len]
    y_test_original = y_test_original[:min_test_len]
    test_preds_original = test_preds_original[:min_test_len]

    plt.figure(figsize=(16, 8))

    if transfer_learning:
        title = f'Transfer Learning LSTM Model - {ticker}\nNormalized RMSE: {metrics["normalized_rmse"]:.4f}, Original RMSE: {metrics["original_rmse"]:.4f}'
    else:
        title = f'LSTM Model - {ticker}\nNormalized RMSE: {metrics["normalized_rmse"]:.4f}, Original RMSE: {metrics["original_rmse"]:.4f}'

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Closing Price USD ($)')

    train_dates_list = train_dates.tolist() if hasattr(train_dates, 'tolist') else list(train_dates)
    test_dates_list = test_dates.tolist() if hasattr(test_dates, 'tolist') else list(test_dates)
    all_dates = train_dates_list + test_dates_list

    y_train_np = y_train_original.numpy() if hasattr(y_train_original, 'numpy') else y_train_original
    y_test_np = y_test_original.numpy() if hasattr(y_test_original, 'numpy') else y_test_original

    y_train_list = y_train_np.flatten().tolist() if hasattr(y_train_np, 'flatten') else list(y_train_np)
    y_test_list = y_test_np.flatten().tolist() if hasattr(y_test_np, 'flatten') else list(y_test_np)

    all_real_values = y_train_list + y_test_list
    plt.plot(all_dates, all_real_values, color='gray', label='Real Data')

    train_preds_np = train_preds_original.numpy() if hasattr(train_preds_original, 'numpy') else train_preds_original
    test_preds_np = test_preds_original.numpy() if hasattr(test_preds_original, 'numpy') else test_preds_original

    train_preds_flat = train_preds_np.flatten() if hasattr(train_preds_np, 'flatten') else train_preds_np
    test_preds_flat = test_preds_np.flatten() if hasattr(test_preds_np, 'flatten') else test_preds_np

    if transfer_learning:
        all_pred_dates = train_dates_list + test_dates_list
        all_pred_values = list(train_preds_flat) + list(test_preds_flat)
        plt.plot(all_pred_dates, all_pred_values, color='skyblue', label='Predicted Data')
    else:
        plt.plot(train_dates, train_preds_flat, color='skyblue', label='Predicted - Train')
        plt.plot(test_dates, test_preds_flat, color='blue', label='Predicted - Test')

    plt.legend()
    plt.grid(True, alpha=0.3)

    if save:
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)

        filename = f"{ticker}_transfer_learning.png" if transfer_learning else f"{ticker}_simple_training.png"
        plt.savefig(os.path.join(results_dir, filename))

    plt.show()


def calculate_additional_metrics(predictions, full=False):
    """
    Calculate additional performance metrics from model predictions.

    Args:
        predictions (tuple): Tuple containing (train_preds_original, test_preds_original,
                            y_train_original, y_test_original)
        full (bool, optional): If True, combine train and test data and store in 'test' key,
                              with 'train' set to None. Defaults to False.

    Returns:
        dict: Dictionary of calculated metrics for both training and test sets,
              or just combined data if full=True
    """
    train_preds, test_preds, y_train, y_test = predictions

    # Ensure all are numpy arrays and flattened
    train_preds = np.array(train_preds).flatten()
    test_preds = np.array(test_preds).flatten()
    y_train = np.array(y_train).flatten()
    y_test = np.array(y_test).flatten()

    # Dictionary to store all metrics
    metrics = {'train': {}, 'test': {}}

    # Define data sets to process
    if full:
        # Combine train and test data
        data_sets = {
            'test': (np.concatenate([train_preds, test_preds]), np.concatenate([y_train, y_test]))
        }
        metrics['train'] = None
    else:
        data_sets = {
            'train': (train_preds, y_train),
            'test': (test_preds, y_test)
        }

    # Calculate metrics for each data set
    for set_name, (preds, actual) in data_sets.items():
        # 1. Mean Absolute Error (MAE)
        metrics[set_name]['mae'] = mean_absolute_error(actual, preds)

        # 2. Mean Absolute Percentage Error (MAPE)
        metrics[set_name]['mape'] = np.mean(np.abs((actual - preds) / actual)) * 100

        # 3. R-squared (R²)
        metrics[set_name]['r2'] = r2_score(actual, preds)

        # 4. Direction Accuracy
        direction_actual = np.sign(actual[1:] - actual[:-1])
        direction_pred = np.sign(preds[1:] - preds[:-1])
        metrics[set_name]['direction_accuracy'] = np.mean(direction_actual == direction_pred) * 100

        # 5. Maximum Error
        metrics[set_name]['max_error'] = np.max(np.abs(actual - preds))

        # 6. Simple Trading Simulation
        metrics[set_name]['strategy_return'] = simulate_trading(actual[1:], direction_pred)
        metrics[set_name]['buy_hold_return'] = (actual[-1] / actual[0] - 1) * 100

    return metrics

def simulate_trading(prices, predicted_directions):
    """
    Simple trading simulation based on predicted directions.

    Args:
        prices (array): Actual price series
        predicted_directions (array): Predicted direction (1 for up, -1 for down, 0 for no change)

    Returns:
        float: Percentage return from the trading strategy
    """
    position = 0  # 0 = no position, 1 = long
    cash = 100  # Starting with $100
    shares = 0

    for i in range(len(prices) - 1):
        # If predicted up and no position, buy
        if predicted_directions[i] > 0 and position == 0:
            shares = cash / prices[i]
            cash = 0
            position = 1
        # If predicted down and have position, sell
        elif predicted_directions[i] < 0 and position == 1:
            cash = shares * prices[i]
            shares = 0
            position = 0

    # Close any remaining position at the end
    if position == 1:
        cash = shares * prices[-1]

    # Calculate return percentage
    return (cash - 100) / 100 * 100


def print_metrics_report(ticker, metrics):
    """
    Print a formatted report of all metrics.

    Args:
        ticker (str): Stock ticker symbol
        metrics (dict): Dictionary of calculated metrics
    """
    print(f"\n{'=' * 50}")
    print(f"ADDITIONAL METRICS REPORT FOR {ticker}")
    print(f"{'=' * 50}")

    if metrics['train'] is not None:
        print("\nTraining Set Metrics:")
        print(f"Mean Absolute Error (MAE): {metrics['train']['mae']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['train']['mape']:.2f}%")
        print(f"R-squared (R²): {metrics['train']['r2']:.4f}")
        print(f"Direction Accuracy: {metrics['train']['direction_accuracy']:.2f}%")
        print(f"Maximum Error: ${metrics['train']['max_error']:.4f}")
        print(f"Strategy Return: {metrics['train']['strategy_return']:.2f}%")
        print(f"Buy & Hold Return: {metrics['train']['buy_hold_return']:.2f}%")

    print("\nTest Set Metrics:")
    print(f"Mean Absolute Error (MAE): ${metrics['test']['mae']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['test']['mape']:.2f}%")
    print(f"R-squared (R²): {metrics['test']['r2']:.4f}")
    print(f"Direction Accuracy: {metrics['test']['direction_accuracy']:.2f}%")
    print(f"Maximum Error: ${metrics['test']['max_error']:.4f}")
    print(f"Strategy Return: {metrics['test']['strategy_return']:.2f}%")
    print(f"Buy & Hold Return: {metrics['test']['buy_hold_return']:.2f}%")