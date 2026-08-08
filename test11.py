import pandas as pd

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

df = pd.read_csv('test_FD001.txt', names=column_labels(), header=None, sep='\s+')


print(df['time_cycles'].describe())

