import os
import datetime
import sys
import pandas as pd
from data_preprocessing import preprocess_data, create_date_features, scale_data, split_data
from data_helper import result_path, dataset_name, file_list, features_path
from lstm_model import build_lstm_model
from output_writer import OutputWriter

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

        # Scale the data
        scaler, X_scaler, y_scaler = scale_data(df)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, X_scaler, y_scaler)
        print("The number of training samples: ", X_train.shape[0])
        print("The number of validation samples: ", X_val.shape[0])
        print("The number of testing samples: ", X_test.shape[0])

        print("Starting training...")
        # Train/Predict LSTM model
        model = build_lstm_model(X_train, y_train, X_val, y_val)
        print("Testing model")
        ytest_unscaled_prediction = model.predict(X_test)

        # Inverse transform
        ytest_prediction = scaler.inverse_transform(ytest_unscaled_prediction)
        ytest_ground_truth = scaler.inverse_transform(y_test)
        y_train = scaler.inverse_transform(y_train)
        y_val = scaler.inverse_transform(y_val)

        # Write output
        output_writer.write_output_dl_results(y_train=y_train,
                                              y_val=y_val,
                                              ground_truth=ytest_ground_truth,
                                              predictions=ytest_prediction,
                                              model_name="LSTM",
                                              indent=2)


if __name__ == '__main__':
    main()
