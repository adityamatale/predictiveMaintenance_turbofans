import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Simulate Defense Data
# Notice the relationships I'm creating:
# - Faster speed -> More fuel burn (Positive correlation)
# - Higher altitude -> Lower temperature (Negative correlation)
# - Pilot ID -> Random number (No correlation)
data = {
    'speed_mach': [0.8, 1.2, 1.5, 0.9, 2.0],
    'fuel_burn_rate': [50, 80, 110, 55, 150],     # Correlated with speed
    'altitude_ft': [1000, 15000, 25000, 5000, 40000],
    'air_temp_c': [25, -10, -35, 15, -55],        # Correlated with altitude
    'pilot_id': [101, 204, 305, 108, 550]         # Just a label (Noise)
}

df_jet = pd.DataFrame(data)

# 2. Calculate the Matrix
corr_matrix = df_jet.corr()

# 3. Visualize it (Standard Industry Practice)
# In interviews, always mention "Heatmaps" when talking about correlation.
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Jet Fighter Sensor Correlation Matrix")
plt.show()

# --- Interpreting the Output ---
# You will see 'speed_mach' and 'fuel_burn_rate' have a score near 1.0 (Red).
# You will see 'altitude_ft' and 'air_temp_c' have a score near -1.0 (Blue).
# 'pilot_id' will have low scores (near 0) with everything else.