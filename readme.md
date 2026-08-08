# Predictive Maintenance for Turbofan Engines

A machine learning project for predicting the **Remaining Useful Life (RUL)** of turbofan engines using sensor data from the **NASA C-MAPSS dataset**.

The goal is to identify degradation patterns and estimate how many operational cycles an engine has left before failure.

## Overview

The project follows a simple predictive maintenance workflow:

```text
NASA C-MAPSS Data
        ↓
Data Preprocessing
        ↓
Sensor & Correlation Analysis
        ↓
Feature Selection
        ↓
RUL Generation
        ↓
Model Training
        ↓
Model Evaluation
```

## Dataset

The project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset**.

Four subsets are included:

* FD001
* FD002
* FD003
* FD004

Each dataset contains engine operating conditions and multiple sensor measurements recorded across operational cycles.

The target variable is **Remaining Useful Life (RUL)**.

## Machine Learning Models

The project experiments with multiple regression approaches:

* **Linear Regression**
* **Random Forest Regressor**
* **XGBoost Regressor**

The models are trained to estimate the remaining operational cycles of each engine.

## Analysis & Preprocessing

The preprocessing pipeline includes:

* Sensor data exploration
* Correlation analysis
* Feature selection
* Data cleaning
* RUL calculation
* Train/test dataset preparation

## Evaluation

Model performance is evaluated using standard regression metrics:

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **R² Score** — Coefficient of Determination

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn**


## Getting Started

Clone the repository:

```bash
git clone https://github.com/adityamatale/predictiveMaintenance_turbofans.git
cd predictiveMaintenance_turbofans
```

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

Run the desired model script:

```bash
python xgtboost.py
```

Other model implementations can be run similarly.

## Key Takeaway

This project demonstrates how **machine learning and sensor data can be used for predictive maintenance**, helping estimate equipment degradation and identify potential failures before they occur.
