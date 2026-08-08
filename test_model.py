from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

##Cleaning Data
def get_dead_sensors(dataframe: pd.DataFrame)->list[str]:
    """
    Identifies columns that have a single constant value (Variance = 0).

    :param dataframe: Pass a Data frame so that we can identify all columns.
    :return: A list of the column names that have a single constant value.
    """
    n_unique = dataframe.nunique()
    dead_sensors = n_unique[n_unique == 1].index.tolist()
    #The output returns the following list ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19"]
    return dead_sensors

def column_labels():
    """
    This takes no variables and is only custom-made for this dataset

    Gives a list of the column names that can be used within the Data Frame

    :return: A list of column names
    """
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    return col_names

df = pd.read_csv('test_FD001.txt', sep='\s+', names=column_labels(), header=None)

new_df = df.drop(columns=get_dead_sensors(df))
new_df.to_csv('test_FD001_Processed.csv', index=False)

df = pd.read_csv('test_FD001_Processed.csv')
X_test_final = df.groupby('unit_number').last().reset_index()

features = [col for col in df.columns if col not in ['unit_number', 'time_cycles']]

X_test_final = X_test_final[features]
Y_test_final = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None, names=['RUL'])
Y_test_clipped = Y_test_final.clip(upper=125)

model = RandomForestRegressor(n_estimators=100, random_state=50, max_depth=15)
prediction = model.predict(X_test_final)

final_rmse = np.sqrt(mean_squared_error(Y_test_clipped, prediction))
print(final_rmse)




