from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('processed_train_FD001.csv')

features = [col for col in df.columns if col not in ['unit_number', 'time_cycles', 'RUL']]
target = 'RUL'

train_df = df[df['unit_number'] <= 75]
test_df  = df[df['unit_number'] > 75]

X_train = train_df[features]
y_train = train_df[target].clip(upper=125)

X_test  = test_df[features]
y_test  = test_df[target].clip(upper=125)


xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=50,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)


y_pred_xgb = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb  = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb   = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb:.2f} cycles")
print(f"XGBoost MAE : {mae_xgb:.2f} cycles")
print(f"XGBoost R²  : {r2_xgb:.3f}")


xgb_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 8))
sns.barplot(
    data=xgb_importance_df.head(20),
    x='Importance',
    y='Feature',
    palette='viridis'
)

plt.title('Top 20 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Sensor Feature')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.4)
plt.plot([0, 125], [0, 125], linestyle='--')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')
plt.title('True vs Predicted RUL (XGBoost)')
plt.tight_layout()
plt.show()


# results_df = pd.DataFrame({
#     'Model': ['Random Forest', 'XGBoost'],
#     'RMSE': [rmse, rmse_xgb]
# })

# print(results_df)
