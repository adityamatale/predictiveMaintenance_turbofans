#This is a new baseline model, the first one using linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv('processed_train_FD001.csv')

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

model = LinearRegression()
model.fit(X_train, Y_clipped_train)

Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_clipped_test, Y_pred)) #RMSE (Root Mean Squared Error)
print(f"Baseline RMSE: {rmse:.2f} cycles")

#Output was "Baseline RMSE: 50.63 cycles"
