import os
import pandas as pd
from utils import check_dir_exists
from data_helper import data_path, features_path
from data_preprocessing import preprocess_tsdata, create_date_features, create_lagged_features

import warnings

warnings.filterwarnings('ignore')


# Code
def main():
    ts_data = pd.read_csv(data_path)
    df = preprocess_tsdata(ts_data)
    df = df.set_index('date')

    # Check folder exits
    check_dir_exists(features_path)

    # Generate a new dataset with new features ['day', 'month', 'year', 'quarter', 'dayofweek', 'dayofyear']
    date_df = df.copy()
    date_df = create_date_features(date_df)
    date_df.to_csv(os.path.join(features_path, 'date_smedebtsu.csv'))
    print("File saved successfully!")
    print("File path: ", os.path.join(features_path, 'date_smedebtsu.csv'))

    # Generate a new dataset with lagged features ['total_debts_lag1', 'total_debts_lag2']
    debt_df = df.copy()
    debt_df = create_lagged_features(debt_df)
    debt_df.to_csv(os.path.join(features_path, 'lag_smedebtsu.csv'))
    print("File saved successfully!")
    print("File path: ", os.path.join(features_path, 'lag_smedebtsu.csv'))

    return date_df, debt_df


if __name__ == '__main__':
    main()
