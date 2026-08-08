from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

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

model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=50)
model.fit(X_train, Y_clipped_train)

Y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_clipped_test, Y_pred))
print(f"Baseline RMSE: {rmse:.2f} cycles")

####################>>>>Report<<<<#######################

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

