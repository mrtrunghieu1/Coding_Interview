import os
import datetime
import sys
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_helper import features_path, result_path, dataset_name, file_list
from data_helper import N_SPLITS, MAX_TRAIN_SIZE, TEST_SIZE, NUM_OF_TEST
from data_preprocessing import preprocess_data
from output_writer import OutputWriter
from utils import compute_mean_metric

import warnings

warnings.filterwarnings('ignore')

# Code
try:
    # Run with file_list[from_id, from_id + 1, ..., to_id - 1]
    from_id = int(sys.argv[1])
    to_id = int(sys.argv[2])
except:
    from_id = 0
    to_id = len(file_list)


def main():
    for i_file in range(from_id, to_id):
        file_name = file_list[i_file]
        print(datetime.datetime.now(), " File {}: {}".format(i_file, file_name))
        output_writer = OutputWriter(result_path, file_name.split('.')[0])

        # Read file csv
        dataset_path = os.path.join(features_path, file_name)
        df = pd.read_csv(dataset_path)
        df = preprocess_data(df)
        df = df.set_index('date')

        # Regression Algorithms
        regressors = {
            "linear_regression": LinearRegression(),
            "xgboost_regressor": xgb.XGBRegressor(),
            "light_GBM": LGBMRegressor(),
            "random_forest_regressor": RandomForestRegressor(),
        }
        # Dict
        ml_prediction_dict = {
            "linear_regression": [],
            "xgboost_regressor": [],
            "light_GBM": [],
            "random_forest_regressor": []
        }

        ml_test_dict = {
            "linear_regression": [],
            "xgboost_regressor": [],
            "light_GBM": [],
            "random_forest_regressor": []
        }

        # Train/ Test Split
        tss = TimeSeriesSplit(n_splits=N_SPLITS, max_train_size=MAX_TRAIN_SIZE, test_size=TEST_SIZE)
        for fold_num, (train_index, test_index) in enumerate(tss.split(df)):
            train = df.iloc[train_index]
            test = df.iloc[test_index]

            TARGET = 'total_debts'
            FEATURES = [feature for feature in df.columns if feature != TARGET]

            X_train, y_train = train[FEATURES], train[TARGET]
            X_test, y_test = test[FEATURES], test[TARGET]

            for name_regressor, reg in regressors.items():
                reg.fit(X_train, y_train)
                y_prediction = reg.predict(X_test)
                # -------------------------------------------------------------------------------------
                mae = mean_absolute_error(y_test, y_prediction)
                mse = mean_squared_error(y_test, y_prediction)
                rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
                r2 = r2_score(y_test.tolist(), y_prediction)
                # ---------------------------- Writing Output -----------------------------------------
                metrics = {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2_score": r2
                }

                predictions = {
                    "date": y_test.index.strftime('%Y-%m-%d').tolist(),
                    "num_test_samples": len(y_test),
                    "target": y_test.tolist(),
                    "predictions": y_prediction.tolist()
                }
                ml_test_dict[name_regressor].append(y_test.tolist())
                ml_prediction_dict[name_regressor].append(y_prediction.tolist())
                output_writer.write_fold_results(fold_num + 1, name_regressor, metrics, predictions, indent=2)

        # Calculate the mean of MAE, MSE, RMSE, R2_score across all folds for each regression algorithms
        final_output_list = []
        rmse_benchmark = {}
        for name_regressor in regressors.keys():
            metrics_list = []
            for fold_num in range(1, N_SPLITS + 1):
                fold_results = output_writer.get_results[name_regressor][f"Fold_{fold_num}"]["metrics"]
                metrics_list.append(fold_results)

            mean_mae = compute_mean_metric(metrics_list, N_SPLITS, method="MAE")
            mean_mse = compute_mean_metric(metrics_list, N_SPLITS, method="MSE")
            mean_rmse = compute_mean_metric(metrics_list, N_SPLITS, method="RMSE")
            mean_r2 = compute_mean_metric(metrics_list, N_SPLITS, method="R2_score")

            # ---------------------------- Writing Output -----------------------------------------
            final_output = {
                "dataset": file_name.split('.')[0],
                "K_folds": N_SPLITS,
                "model": name_regressor,
                "num_test_samples": NUM_OF_TEST,
                "mean_MAE": mean_mae,
                "mean_MSE": mean_mse,
                "mean_RMSE": mean_rmse,
                "mean_R2_score": mean_r2,
                "test": list(itertools.chain(*ml_test_dict[name_regressor])), # convert 2d to 1d
                "prediction": list(itertools.chain(*ml_prediction_dict[name_regressor]))
            }
            final_output_list.append(final_output)
            rmse_benchmark.update({name_regressor: mean_rmse})
        output_writer.write_average_results(final_output_list, indent=2)


if __name__ == '__main__':
    main()
