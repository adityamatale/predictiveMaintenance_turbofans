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
    #The output returns the following list ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19"] For 001
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

#### Values that need to be changed for testing

df_filename = 'train_FD004.txt'
processed_filename = 'processed_train_FD004.csv'

####

df = pd.read_csv(df_filename, sep='\s+', names=column_labels(), header=None)

#Getting the maximum cycles and using that to get RUL
max_cycles = df.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycles']
df = df.merge(max_cycles, on='unit_number', how='left')
df['RUL'] = df['max_cycles'] - df['time_cycles']
df = df.drop(columns=['max_cycles'], errors='ignore')

#Dropping any dead sensors
new_df = df.drop(columns=get_dead_sensors(df))

#Saving the new data frame we retrieved
#new_df = new_df.drop(columns=['setting_3', 's_19', 's_5', 's_18', 's_16', 's_10', 's_9', 's_8','s_2','s_3','s_7','s_6','s_21','s_20','s_17','setting_2','setting_1','s_1'])
new_df.to_csv(processed_filename, index=False)

#Creating a random forest model
df = pd.read_csv(processed_filename)

features = [col for col in df.columns if col not in ['unit_number', 'time_cycles', 'RUL']]
target = 'RUL'

training_unit_numbers = df[df['unit_number'] <= 75]
testing_unit_numbers = df[df['unit_number'] > 75]

X_train = training_unit_numbers[features]
Y_train = training_unit_numbers[target]

X_test = testing_unit_numbers[features]
Y_test = testing_unit_numbers[target]

Y_clipped_train = Y_train.clip(upper=125)
Y_clipped_test = Y_test.clip(upper=125)

#passing the values to the model to predict
model = RandomForestRegressor(n_estimators=100, random_state=45, max_depth=15)
model.fit(X_train, Y_clipped_train)

Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_clipped_test, Y_pred)) #RMSE (Root Mean Squared Error)
print(f"Baseline RMSE: {rmse:.2f} cycles")

####>>>> Report <<<<####

importances = model.feature_importances_
feature_names = features

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x='Feature', y='Importance', hue='Feature', data=importance_df, palette='viridis')

plt.title('Which feature was used with what importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

#### Running actual test set ####

df = pd.read_csv('test_FD004.txt', sep='\s+', names=column_labels(), header=None)

new_df = df.drop(columns=get_dead_sensors(df))
#new_df = new_df.drop(columns=['setting_3', 's_19', 's_5', 's_18', 's_16', 's_10', 's_9', 's_8','s_2','s_3','s_7','s_6','s_21','s_20','s_17','setting_2','setting_1','s_1'])
new_df.to_csv('test_FD004_Processed.csv', index=False)

df = pd.read_csv('test_FD004_Processed.csv')
X_test_final = df.groupby('unit_number').last().reset_index()

features = [col for col in df.columns if col not in ['unit_number', 'time_cycles']]

X_test_final = X_test_final[features]
Y_test_final = pd.read_csv('RUL_FD004.txt', sep='\s+', header=None, names=['RUL'])
Y_test_clipped = Y_test_final.clip(upper=125)

prediction = model.predict(X_test_final)

final_rmse = np.sqrt(mean_squared_error(Y_test_clipped, prediction))
print("Final Test Set RMSE :", final_rmse)

####>>>> Report <<<<####

importances = model.feature_importances_
feature_names = features

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x='Feature', y='Importance', hue='Feature', data=importance_df, palette='viridis')

plt.title('Which feature was used with what importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()



