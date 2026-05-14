# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:55:20 2026

@author: 2577
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

###############################################################################
# Step1: simulate bike demand data
###############################################################################
np.random.seed(123)
n = 500
hour = np.random.randint(0, 24, n)
temp = np.random.uniform(0, 35, n)  # Celsius
is_weekend = np.random.binomial(1, 0.3, n)

# Demand model (heteroscedastic)
demand = (
    50 
    + 3 * temp 
    + 10 * (hour >= 7) * (hour <= 9)  # Morning rush
    + 15 * (hour >= 17) * (hour <= 19)  # Evening rush
    + 20 * is_weekend  # Weekends higher
    + np.random.normal(0, 5 + 0.5 * temp)  # Variance grows with temp
)
demand = np.maximum(demand, 0)  # No negative rentals

df_bikes = pd.DataFrame({
    'temp': temp,
    'hour': hour,
    'is_weekend': is_weekend,
    'demand': demand
})

# Split train/test
from sklearn.model_selection import train_test_split

# 80/20 split is standard for n > 500. For smaller data (n < 200), use 70/30 or cross-validation.
train, test = train_test_split(df_bikes, test_size=0.2, random_state=42)


###############################################################################
# Step2: fit Multi-Quantile Model
###############################################################################
X_train = train[['temp', 'hour', 'is_weekend']].values
y_train = train['demand'].values
X_test = test[['temp', 'hour', 'is_weekend']].values
y_test = test['demand'].values

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit quantiles
quantiles_bike = [0.1, 0.5, 0.9]
models_bike = {}
preds_test = {}

for q in quantiles_bike:
    model = sm.QuantReg(y_train, X_train_const).fit(q=q)
    models_bike[q] = model
    preds_test[q] = model.predict(X_test_const)
    print(f"τ={q} | β_temp={model.params[1]:.2f}, β_hour={model.params[2]:.2f}, β_weekend={model.params[3]:.2f}")


###############################################################################
## Step3: evaluate Coverage on Test Set
###############################################################################
lower_test = preds_test[0.1]
upper_test = preds_test[0.9]
in_interval_test = (y_test >= lower_test) & (y_test <= upper_test)
coverage_test = in_interval_test.mean()

print(f"Test set coverage (80% PI): {coverage_test:.2%}")

###############################################################################
## Step4: standarization
###############################################################################
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


###############################################################################
## Step6: residual Patterns
###############################################################################
residuals_median = y_test - preds_test[0.5]
plt.scatter(preds_test[0.5], residuals_median, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual Plot (Median QR)')
plt.show()

###############################################################################
###############################################################################
## LightGBM
###############################################################################
###############################################################################

import lightgbm as lgb
params = {
    'objective': 'quantile',  # Pinball loss
    'alpha': 0.9,             # τ (target quantile)
    'metric': 'quantile',     # Evaluation metric
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_data_in_leaf': 50,
    'verbose': -1
}
dtrain = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, dtrain, num_boost_round=500)


###############################################################################
## Step 1: Generate Non-Linear Data
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n = 1000
# Non-linear relationship with heteroscedasticity
X = np.random.uniform(0, 4, n)
y = np.sin(2 * X) + 0.3 * X**2 + np.random.normal(0, 0.5 + 0.3 * X**2)
df = pd.DataFrame({'X': X, 'y': y})
# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['y'], alpha=0.5, s=20)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-Linear Data with Heteroscedasticity')
plt.show()

###############################################################################
## Step 2: Train-Test Split
###############################################################################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df[['X']].values, df['y'].values, test_size=0.2, random_state=42
)

###############################################################################
## Step 3: Fit Quantile Models with LightGBM
###############################################################################

import lightgbm as lgb
quantiles = [0.1, 0.5, 0.9]
models = {}
for tau in quantiles:
    params = {
        'objective': 'quantile',
        'alpha': tau,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'seed': 42
    }
    try:
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=300)
        models[tau] = model
        print(f"Trained model for τ={tau}")
    except Exception as e:
        print(f"Error training τ={tau}: {e}")
        print("Tip: Try reducing num_boost_round or increasing min_data_in_leaf")
        raise

###############################################################################
## Step 4: Predict and Visualize
###############################################################################

# Sort X for plotting
X_plot = np.linspace(0, 4, 200).reshape(-1, 1)
predictions = {tau: models[tau].predict(X_plot) for tau in quantiles}
plt.figure(figsize=(12, 7))
plt.scatter(X_test, y_test, alpha=0.3, s=20, label='Test data', color='gray')
colors = {0.1: 'blue', 0.5: 'green', 0.9: 'red'}
for tau in quantiles:
    plt.plot(X_plot, predictions[tau], color=colors[tau], lw=2.5, 
             label=f'QR(τ={tau})')
# Shade 80% prediction interval
plt.fill_between(X_plot.flatten(), predictions[0.1], predictions[0.9], 
                 alpha=0.2, color='orange', label='80% PI')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Non-Linear Quantile Regression with LightGBM', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

###############################################################################
## Multi-Feature Example
###############################################################################


# Generate data with multiple features
np.random.seed(123)
n = 2000
df_multi = pd.DataFrame({
    'temp': np.random.uniform(0, 40, n),
    'humidity': np.random.uniform(20, 100, n),
    'hour': np.random.randint(0, 24, n),
    'is_weekend': np.random.binomial(1, 0.3, n)
})

# Target: demand with non-linear effects + heteroscedasticity
df_multi['demand'] = (
    50
    + 2 * df_multi['temp']
    + 0.3 * df_multi['humidity']
    + 10 * (df_multi['hour'] >= 17) * (df_multi['hour'] <= 20)  # Rush hour
    + 15 * df_multi['is_weekend']
    + np.random.normal(0, 5 + 0.2 * df_multi['temp'])  # Variance grows with temp
)

X = df_multi[['temp', 'humidity', 'hour', 'is_weekend']].values
y = df_multi['demand'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



feature_names = ['temp', 'humidity', 'hour', 'is_weekend']
quantiles_multi = [0.1, 0.5, 0.9]
importances = {}

for tau in quantiles_multi:
    params = {
        'objective': 'quantile',
        'alpha': tau,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=300)
    # Normalize importance to sum to 100% for each quantile
    raw_importance = model.feature_importance(importance_type='gain')
    importances[tau] = 100 * raw_importance / raw_importance.sum()

# Visualize
import pandas as pd

df_importance = pd.DataFrame(importances, index=feature_names)
print(df_importance)

df_importance.T.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Quantile', fontsize=12)
plt.ylabel('Feature Importance (%)', fontsize=12)
plt.title('Feature Importance Across Quantiles (Normalized)', fontsize=14)
plt.legend(title='Feature', fontsize=10)
plt.tight_layout()
plt.show()

'''
# Median model
if temp_importance_median < 0.55:  # Down from 0.61
    alert("Median model drift: temperature importance dropped")

# Tail models  
if humidity_importance_tail < 0.25:  # Down from 0.31-0.34
    alert("Tail model drift: humidity importance dropped")
'''



import optuna

def objective(trial):
    """Optuna objective: minimize pinball loss on validation set."""
    # Suggest hyperparameters
    params = {
        'objective': 'quantile',
        'alpha': 0.9,  # Fix quantile (tune per quantile separately)
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1,
        'seed': 42
    }
    
    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    model = lgb.train(params, dtrain, num_boost_round=300)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Compute pinball loss
    tau = params['alpha']
    residual = y_val - y_pred
    loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()
    
    return loss





# Create study
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

# Optimize (50 trials ~ 5 minutes)
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Best parameters
print("Best hyperparameters:")
print(study.best_params)
print(f"Best validation loss: {study.best_value:.4f}")


best_params = {
    'objective': 'quantile',
    'alpha': 0.9,
    **study.best_params,
    'verbose': -1
}

dtrain_full = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(best_params, dtrain_full, num_boost_round=500)

# Evaluate on test set
y_pred_test = final_model.predict(X_test)
residual_test = y_test - y_pred_test
test_loss = np.where(residual_test >= 0, 0.9 * residual_test, -0.1 * residual_test).mean()
print(f"Test pinball loss: {test_loss:.4f}")


from optuna.visualization.matplotlib import plot_optimization_history
import matplotlib.pyplot as plt
import optuna

# Optimization progress
plot_optimization_history(study)
plt.tight_layout()
plt.show()

# Parameter importance (manual, avoids set_yticks compatibility bug)
importances = optuna.importance.get_param_importances(study)
fig, ax = plt.subplots()
params = list(importances.keys())
values = list(importances.values())
ax.barh(params, values)
ax.set_xlabel("Importance")
ax.set_title("Hyperparameter Importances")
plt.tight_layout()
plt.show()





def optimize_for_quantile(tau, n_trials=30):
    """Run Bayesian optimization for a specific quantile."""
    def objective(trial):
        params = {
            'objective': 'quantile',
            'alpha': tau,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
            'verbose': -1,
            'seed': 42
        }
        
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model = lgb.train(params, dtrain, num_boost_round=300)
        
        y_pred = model.predict(X_val)
        residual = y_val - y_pred
        loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()
        return loss
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

# Tune each quantile
best_params_by_quantile = {}
for tau in [0.1, 0.5, 0.9]:
    print(f"\nOptimizing for τ={tau}...")
    best_params_by_quantile[tau] = optimize_for_quantile(tau, n_trials=30)
    print(f"Best params: {best_params_by_quantile[tau]}")


# For each sample, enforce monotonicity across quantiles
# Example: if sample i has Q_0.5 = 10 and Q_0.9 = 9 (crossing),
# sorting gives Q_0.5 = 9, Q_0.9 = 10 (fixed)
preds_corrected = {tau: [] for tau in taus}
for i in range(len(X_test)):
    quantile_vals = np.array([preds[tau][i] for tau in taus])
    quantile_vals_sorted = np.sort(quantile_vals)  # Enforce Q_0.1 ≤ Q_0.5 ≤ Q_0.9
    for j, tau in enumerate(taus):
        preds_corrected[tau].append(quantile_vals_sorted[j])

# Convert to arrays
for tau in taus:
    preds_corrected[tau] = np.array(preds_corrected[tau])
    
    
    
# Weekly batch job
coverage_weekly = (y_actual >= pred_lower) & (y_actual <= pred_upper).mean()
if coverage_weekly < 0.75:  # Trigger alert if < 75% (target: 80%)
    send_alert("QR model miscalibrated: retrain needed")
    
    
# Save
model.save_model(f'qr_tau09_v2.txt')

# Load
loaded_model = lgb.Booster(model_file='qr_tau09_v2.txt')

# Version with metadata
model_metadata = {
    'tau': 0.9,
    'train_date': '2025-10-12',
    'n_train': len(X_train),
    'features': feature_names,
    'test_coverage': 0.89
}
# Save metadata alongside model
    
    
    
    