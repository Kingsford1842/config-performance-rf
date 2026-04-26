import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def train_random_forest(train_X, train_Y, seed):
    """
    Train a Random Forest model using the given training data.
    """
    model = RandomForestRegressor(
        n_estimators=100,     # number of trees
        max_depth=None,       # allow trees to grow
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",  # the random subset of features per split
        random_state=seed,
        n_jobs=-1                 
    )
    model.fit(train_X, train_Y)
    return model


def evaluate(y_true, y_pred):
    """
    Calculate evaluation metrics for predictions.
    """
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mape, mae, rmse


def main():
    # Specify the parameters
    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    num_repeats = 3
    train_frac = 0.7
    random_seed = 1

    for system in systems:
        dataset_path = f'datasets/{system}'
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            print(f"\n> System: {system}, Dataset: {csv_file}")

            # Load the datasets
            data = pd.read_csv(os.path.join(dataset_path, csv_file))

            # Store metrics across repeats
            metrics = {'MAPE': [], 'MAE': [], 'RMSE': []}

            for repeat in range(num_repeats):
                # Changing the seed each time to avoid bias
                seed = random_seed * repeat

                # Splitting into training and testing sets
                train_data = data.sample(frac=train_frac, random_state=seed)
                test_data = data.drop(train_data.index)

                # Separate features and target 
                train_X = train_data.iloc[:, :-1]
                train_Y = train_data.iloc[:, -1]
                test_X = test_data.iloc[:, :-1]
                test_Y = test_data.iloc[:, -1]

                # train the model
                model = train_random_forest(train_X, train_Y, seed)

                # Make predictions
                predictions = model.predict(test_X)

                # Evaluate the predictions
                mape, mae, rmse = evaluate(test_Y, predictions)

                # Store the results
                metrics['MAPE'].append(mape)
                metrics['MAE'].append(mae)
                metrics['RMSE'].append(rmse)

            # Compute average performance across repeats 
            avg_mape = np.mean(metrics['MAPE'])
            avg_mae = np.mean(metrics['MAE'])
            avg_rmse = np.mean(metrics['RMSE'])

            print(f"RF → MAPE: {avg_mape:.4f} | MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f}")


if __name__ == "__main__":
    main()