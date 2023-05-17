import os

# Paths
root_path = r'C:\Users\Admin\Desktop\Code\Coding_Interview'

# The list of files
file_list = ['date_smedebtsu.csv', 'lag_2_smedebtsu.csv', 'lag_4_smedebtsu.csv']

features_path = os.path.join(root_path, 'data/features')
data_path = os.path.join(root_path, 'data/processed/processed_smedebtsu.csv')
result_path = os.path.join(root_path, 'results')

dataset_name = 'smedebtsu'

# Parameters
N_SPLITS = 5
MAX_TRAIN_SIZE = 60
TEST_SIZE = 4
NUM_OF_TEST = N_SPLITS * TEST_SIZE

# Name of metrics
name_metrics = ["MAE", "MSE", "RMSE", "R2_score"]

# Index boundary test set in Deep Learning model
boundary_idx_test = '2020-12-10'

# Deep Learning parameters
TRAIN_SIZE_RATIO = 0.8
EPOCHS = 10
BATCH_SIZE = 16
