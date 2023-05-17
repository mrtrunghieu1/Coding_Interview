import os
import glob
import json
import itertools
from data_helper import name_metrics, boundary_idx_test


def check_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def compute_mean_metric(metrics_list, n_cv_folds, method="RMSE"):
    if method not in name_metrics:
        raise ValueError(f"Invalid method: {method}. Available methods: {name_metrics}")
    mean_metric = sum(element[method] for element in metrics_list) / n_cv_folds
    return mean_metric


def get_y_train(df):
    data_train = df[df.index < boundary_idx_test]
    y_all_train = data_train['total_debts'].tolist()
    return y_all_train


def flatten_list(lst):
    """
    Flattens a nested list into a single-level list.

    Parameters:
        lst (list): The nested list to be flattened.

    Returns:
        list: A flattened version of the input nested list.

    Example:
        #>>> nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        #>>> flattened_list = flatten_list(nested_list)
        #>>> print(flattened_list)
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    return list(itertools.chain(*lst))
