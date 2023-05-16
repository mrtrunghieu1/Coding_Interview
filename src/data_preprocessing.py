import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_helper import boundary_idx_test, TRAIN_SIZE_RATIO


# Code
def preprocess_data(ts_data):
    """
    Creates a new DataFrame called 'df' containing only the 'date' and 'total_debts' columns from 'ts_data'.
    Parameters:
        ts_data (pandas DataFrame): The DataFrame to preprocess

    Returns:
        new_df (pandas DataFrame): The preprocessed DataFrame containing only the 'date' and 'total_debts' columns
    """
    ts_data['date'] = pd.to_datetime(ts_data['date'])
    return ts_data


def preprocess_tsdata(ts_data):
    ts_data['date'] = pd.to_datetime(ts_data['Date_time'])
    ts_data['total_debts'] = ts_data.sum(axis=1, numeric_only=True)
    new_df = ts_data[['date', 'total_debts']]
    return new_df


def create_date_features(data_frame):
    """
    Create new features that represent different date/time features
    Parameter:
        data_frame (pandas DataFrame): The DataFrame to which new columns will be added
    Returns:
        data_frame (pandas DataFrame): The input DataFrame with new date/time features added as columns
    """
    data_frame['day'] = data_frame.index.day
    data_frame['month'] = data_frame.index.month
    data_frame['year'] = data_frame.index.year
    data_frame['quarter'] = data_frame.index.quarter
    data_frame['dayofweek'] = data_frame.index.dayofweek
    data_frame['dayofyear'] = data_frame.index.dayofyear

    return data_frame


def create_lagged_features(data_frame):
    # Create lagged features
    data_frame['total_debts_lag1'] = data_frame['total_debts'].shift(1)
    data_frame['total_debts_lag2'] = data_frame['total_debts'].shift(2)

    # Drop rows with missing values
    data_frame = data_frame.dropna()
    return data_frame


def scale_data(df):
    df_copy = df.copy()

    scaler = MinMaxScaler(feature_range=(0, 1))

    TARGET = 'total_debts'
    FEATURES = [feature for feature in df.columns if feature != TARGET]

    X_scaler = scaler.fit_transform(df_copy[FEATURES])
    y_scaler = scaler.fit_transform(np.asarray(df_copy[TARGET]).reshape(-1, 1))

    return scaler, X_scaler, y_scaler


def split_data(dataframe, X_scaler, y_scaler):
    """
    This function splits the input dataset into training, validation, and test sets based on the specific day.
    and returns the corresponding data arrays.

    Parameters:
        X_scaler:
        y_scaler:
        dataframe:

    Returns:
    """
    # Find the index of the first timestamp that is greater than or equal to boundary_idx_test("24-02-2023")
    dates = dataframe.index
    boundary_idx = dataframe.index.searchsorted(pd.Timestamp(boundary_idx_test))
    # Split the data into training/validation and test sets
    train_val_dates, train_val_X, train_val_y = dates[:boundary_idx], X_scaler[:boundary_idx], y_scaler[:boundary_idx]
    test_dates, X_test, y_test = dates[boundary_idx:], X_scaler[boundary_idx:], y_scaler[boundary_idx:]

    # Further split the training/validation set into the training and validation sets
    train_size = int(len(train_val_X) * TRAIN_SIZE_RATIO)

    train_dates, X_train, y_train = train_val_dates[:train_size], train_val_X[:train_size, :], \
        train_val_y[:train_size, :]
    val_dates, X_val, y_val = train_val_dates[train_size:], train_val_X[train_size:], \
        train_val_y[train_size:, :]

    return X_train, y_train, X_val, y_val, X_test, y_test
