import os
import glob
import json
from data_helper import name_metrics


def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def compute_mean_metric(metrics_list, n_cv_folds, method="RMSE"):
    if method not in name_metrics:
        raise ValueError(f"Invalid method: {method}. Available methods: {name_metrics}")
    mean_metric = sum(element[method] for element in metrics_list) / n_cv_folds
    return mean_metric


def get_metric_files(result_path):
    rmse_benchmark = {
        'dataset': [],
        'LSTM': [],
        'linear_regression': [],
        'xgboost_regressor': [],
        'light_GBM': [],
        'random_forest_regressor': []
    }

    for dataset in os.listdir(result_path):
        dataset_path = os.path.join(result_path, dataset)
        for model_dir in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, model_dir, 'average_results.json')
            with open(file_path) as f:
                data = json.load(f)
                if model_dir == 'DL_models':
                    rmse_benchmark['dataset'].append(data['dataset'])
                    rmse_benchmark['LSTM'].append(data['RMSE'])
                elif model_dir == 'ML_regression_models':
                    for element in data:
                        model_name = element["model"]
                        if model_name in rmse_benchmark:
                            rmse_benchmark[model_name].append(element['mean_RMSE'])

    return rmse_benchmark
