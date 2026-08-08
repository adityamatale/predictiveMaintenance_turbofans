import pandas as pd

index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

train = pd.read_csv('train_FD001.txt', sep='\s+', names=col_names, header=None)


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

"""
new_df = train.drop(columns=get_dead_sensors(train)) #Dropped all the columns with dead sensors and data data
print(new_df.head(1))
print(new_df.columns.tolist())
"""