import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import yfinance as yf


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


def compute_direction_accuracy(y_true, y_pred):
    """
    Compute directional accuracy metrics for stock price movement predictions.

    This function calculates true positives, false positives, true negatives,
    and false negatives based on direction of price movement (up or down).

    Args:
        y_true (array): Original stock prices (ground truth).
        y_pred (array): Predicted stock prices.

    Returns:
        dict: Dictionary containing confusion matrix values and derived metrics:
            - tp: True positives (correctly predicted price increases)
            - fp: False positives (incorrectly predicted price increases)
            - tn: True negatives (correctly predicted price decreases)
            - fn: False negatives (incorrectly predicted price decreases)
            - accuracy: Overall directional accuracy
            - precision: Precision for price increase predictions
            - recall: Recall for price increase predictions
            - f1: F1-score (harmonic mean of precision and recall)
            - specificity: True negative rate
    """
    # Convert to numpy arrays and flatten if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Calculate day-to-day price changes (returns)
    true_changes = np.diff(y_true)
    pred_changes = np.diff(y_pred)

    # Determine directions (1 for up, 0 for down or no change)
    true_direction = (true_changes > 0).astype(int)
    pred_direction = (pred_changes > 0).astype(int)

    # Calculate confusion matrix values
    tp = np.sum((true_direction == 1) & (pred_direction == 1))  # Correctly predicted price increases
    fp = np.sum((true_direction == 0) & (pred_direction == 1))  # Incorrectly predicted price increases
    tn = np.sum((true_direction == 0) & (pred_direction == 0))  # Correctly predicted price decreases/flat
    fn = np.sum((true_direction == 1) & (pred_direction == 0))  # Incorrectly predicted price decreases/flat

    # Calculate metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity)
    }


def plot_confusion_matrix(ticker, y_true, y_pred, save=False, transfer_learning=False):
    """
    Plot a confusion matrix for stock price direction predictions with improved layout.

    Args:
        ticker (str): Stock ticker symbol.
        y_true (array): Original stock prices (ground truth).
        y_pred (array): Predicted stock prices.
        save (bool, optional): Whether to save the plot to a file. Defaults to False.
        transfer_learning (bool, optional): Whether the plot is for a transfer learning model.
                                           Defaults to False.

    Returns:
        dict: Dictionary containing the directional accuracy metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    import os

    metrics = compute_direction_accuracy(y_true, y_pred)

    tp, fp, tn, fn = metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']

    # Create a figure with 2 subplots side by side - one for the matrix, one for metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={'width_ratios': [3, 1]})  # 3:1 ratio

    if transfer_learning:
        title = f'Transfer Learning Direction Prediction - {ticker}\nF1: {metrics["f1"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}'
    else:
        title = f'Direction Prediction - {ticker}\nF1: {metrics["f1"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}'

    fig.suptitle(title, fontsize=14)

    # Create the confusion matrix plot in the first subplot
    cm = np.array([[tn, fp], [fn, tp]])

    # Create a custom colormap that goes from light blue to dark blue
    colors = [(0.9, 0.95, 1.0), (0.0, 0.2, 0.6)]  # Light blue to dark blue
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors, N=100)

    im = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Down', 'Up'])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Down', 'Up'])
    ax1.set_xlabel('Predicted Direction')
    ax1.set_ylabel('Actual Direction')

    # Add text annotations with improved visibility
    for i in range(2):
        for j in range(2):
            # Calculate appropriate text color based on cell value
            cell_value = cm[i, j]
            max_value = cm.max()
            relative_value = cell_value / max_value

            # Use black text for light background, white for dark background
            if relative_value < 0.6:  # Adjusted threshold for better readability
                text_color = "black"
            else:
                text_color = "white"

            ax1.text(j, i, f'{cell_value}',
                     ha="center", va="center",
                     color=text_color,
                     fontweight='bold',  # Make text bold for better visibility
                     fontsize=12)  # Larger font size

    # Display metrics in the second subplot
    ax2.axis('off')  # Turn off axis
    metrics_table = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'],
        'Value': [
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['specificity']:.4f}"
        ]
    }

    # Create a table in the second subplot
    table = ax2.table(
        cellText=[[metrics_table['Metric'][i], metrics_table['Value'][i]] for i in range(len(metrics_table['Metric']))],
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
        bbox=[0.1, 0.4, 0.8, 0.5]  # [left, bottom, width, height]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)  # Make the table larger

    # Add additional information below the table
    pos_count = tp + fn
    neg_count = tn + fp
    total = pos_count + neg_count

    info_text = (f"Total samples: {total}\n"
                 f"Actual up movements: {pos_count} ({pos_count / total:.1%})\n"
                 f"Actual down movements: {neg_count} ({neg_count / total:.1%})")

    ax2.text(0.5, 0.2, info_text,
             ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.5'),
             fontsize=9)

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)  # Make room for the title

    if save:
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)

        filename = f"{ticker}_confusion_matrix_transfer.png" if transfer_learning else f"{ticker}_confusion_matrix.png"
        plt.savefig(os.path.join(results_dir, filename), bbox_inches='tight', dpi=300)

    plt.show()

    return metrics