import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_dead_sensors(df: pd.DataFrame)->list[str]:
    """
    Identifies columns that have a single constant value (Variance = 0).

    :param df: Pass a Data frame so that we can identify all columns.
    :return: A list of the column names that have a single constant value.
    """
    n_unique = df.nunique()
    dead_sensors = n_unique[n_unique == 1].index.tolist()
    #The output returns the following list ['setting_3', 's_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    return dead_sensors

index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

old_df = pd.read_csv('train_FD001.txt', sep='\s+', names=col_names, header=None)

max_cycle = old_df.groupby('unit_number')['time_cycles'].max().reset_index()
max_cycle.columns = ['unit_number', 'max_cycles']
old_df = old_df.merge(max_cycle, on='unit_number', how='left')
old_df['RUL'] = old_df['max_cycles'] - old_df['time_cycles']
old_df = old_df.drop(columns=['max_cycles'], errors='ignore')

new_df = old_df.drop(columns=get_dead_sensors(old_df))

corr = new_df.corr()

plt.figure(figsize=(10, 10))

sns.heatmap(corr, annot=True, cmap="RdYlGn")
plt.title('Sensor Correlation with Remaining Useful Life (RUL)')
plt.show()
print("Positive Correlations")
print(corr['RUL'].sort_values(ascending=False).head(10))
print("Negative Correlations")
print(corr['RUL'].sort_values(ascending=True).head(10))

"""
##############
test = old_df.loc[old_df['unit_number'] == 5]
print(test['time_cycles'].describe())
###############
new_df.to_csv('processed_train_FD001.csv', index=False)
###############
"""

