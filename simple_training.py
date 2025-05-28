import numpy as np
import tensorflow as tf
import os

from tuning import tune_hyperparameters, train_best_model
from utility import plot_results, prepare_stock_data, calculate_additional_metrics, print_metrics_report
from constants import RANDOM_SEED, LOOK_BACK, EPOCHS, TUNING_MAX_TRIALS, TUNING_EXECUTIONS_PER_TRIAL, TICKERS, DEFAULT_TICKER


def run_model(ticker, verbose=True, save_plot=False):
    if verbose:
        print(f"Tuning hyperparameters for {ticker}...")

    tuner, data_info = tune_hyperparameters(
        ticker=ticker,
        look_back=LOOK_BACK,
        max_trials=TUNING_MAX_TRIALS,
        executions_per_trial=TUNING_EXECUTIONS_PER_TRIAL,
        epochs=EPOCHS,
        verbose=verbose
    )

    rmse_normalized, rmse_original, model, predictions = train_best_model(
        tuner=tuner,
        data_splits=data_info[0],
        scaling_info=data_info[1],
        look_back=LOOK_BACK,
        epochs=EPOCHS,
        verbose=verbose
    )

    X_train, y_train, X_test, y_test = data_info[0]

    prepared_data = prepare_stock_data(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2023-01-01",
        look_back=LOOK_BACK,
        verbose=False
    )

    data_dates = prepared_data['dates']

    train_indices = range(LOOK_BACK, LOOK_BACK + len(y_train))
    test_indices = range(LOOK_BACK + len(y_train), LOOK_BACK + len(y_train) + len(y_test))

    train_dates = data_dates[train_indices]
    test_dates = data_dates[test_indices]

    additional_metrics = calculate_additional_metrics(predictions)

    result = {
        'predictions': predictions,
        'dates': (train_dates, test_dates),
        'metrics': {
            'normalized_rmse': rmse_normalized,
            'original_rmse': rmse_original
        },
        'additional_metrics': additional_metrics,
        'model': model
    }

    if verbose:
        plot_results(ticker, result, save=save_plot)
        print_metrics_report(ticker, additional_metrics)

    return result


def run_tests():
    results = {}

    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'=' * 50}")
        print(f"Processing {ticker}")
        print(f"{'=' * 50}")

        result = run_model(ticker, verbose=True, save_plot=True)
        results[ticker] = result

    ranked_tickers = sorted(results.keys(), key=lambda x: results[x]['metrics']['original_rmse'])

    print("\nTickers ranked by RMSE (best to worst):")
    for i, ticker in enumerate(ranked_tickers):
        rmse = results[ticker]['metrics']['original_rmse']
        normalized_rmse = results[ticker]['metrics']['normalized_rmse']
        print(f"{i + 1}. {ticker}: Original RMSE={rmse:.4f}, Normalized RMSE={normalized_rmse:.4f}")

    return results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    run_tests()