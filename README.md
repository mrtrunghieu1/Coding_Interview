# Coding_Interview
Coding interviews test candidate's technical knowledge, coding ability

## Description
The steps process a Time Series problem
1. Exploratory Data Analysis (EDA)
2. Data generation (create 3 new datasets)
3. Building models

- **Statistical models**: Autoregressive Integrated Moving Average (ARIMA)
- **Machine Learning Regression**:  Linear Regression, XGBoost, RandomForest, LightGBM
- **Deep Learning models**: LSTM
4. Visualize the results
5. Hypothesis test: Friedman test and Nemenyi test

## Task Reports
Welcome to the Task Reports folder of my project! This folder contains PDF files with detailed reports on the tasks 
completed throughout the development process.
### Accessing the Task Reports
To access the task reports, follow these steps:
1. Go to the **task_reports** folder in the repository.
2. You will find individual PDF files for each task in the assignment.
3. Click on the PDF file corresponding to the report you want to view.

## Installation
1. Clone this repository to your local machine using the following command:
```bash
git clone https://github.com/mrtrunghieu1/Coding_Interview.git
```
2. Navigate to the project directory:
```bash
cd Coding_Interview
```
3. Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## How to run
### 2.1 Data Exploration
1. Start Jupyter Notebook by executing the following command in your terminal:
```bash
jupyter notebook
```
2. Your web browser should open with the Jupyter Notebook interface.
3. Navigate to the project directory in the Jupyter Notebook interface and open the **Explore_Data_Task_Report.ipynb** file.

### 2.2 Predictive Model
1. Start Jupyter Notebook by executing the following command in your terminal:
```bash
jupyter notebook
```
2. Your web browser should open with the Jupyter Notebook interface.
3. Navigate to the project directory in the Jupyter Notebook interface and open the **Predictive_Model_Task_Report.ipynb** file.

## File Description
The file description provides an overview of the purpose and functionality of each file within the source code of the 
project.
```commandline
- data_generation.py: Runnable code to generate new data
- data_helper.py: Storage of parameters, paths, and constants
- data_preprocessing.py: Preprocessing functions for related data
- demo_run_base_regressor_models.py: Execution of Machine Learning regression models 
- demo_run_dl_model.py: Execution of Deep Learning model
- lstm_model: Implementation of the LSTM model
- output_writer.py: Structure and storage of the results
- utils.py: Utility functions used throughout the project
```
