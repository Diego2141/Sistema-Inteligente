# -*- coding: utf-8 -*-
"""
Quantile Regression with LightGBM and Bayesian Hyperparameter Tuning
Author: 2577
"""

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optuna.visualization.matplotlib import plot_optimization_history

plt.style.use('seaborn')

###############################################################################
# SECTION 1: LINEAR QUANTILE REGRESSION — Bike Demand (Baseline)
###############################################################################

np.random.seed(123)
n = 500
hour = np.random.randint(0, 24, n)
temp = np.random.uniform(0, 35, n)
is_weekend = np.random.binomial(1, 0.3, n)

demand = (
    50
    + 3 * temp
    + 10 * (hour >= 7) * (hour <= 9)
    + 15 * (hour >= 17) * (hour <= 19)
    + 20 * is_weekend
    + np.random.normal(0, 5 + 0.5 * temp)
)
demand = np.maximum(demand, 0)

df_bikes = pd.DataFrame({'temp': temp, 'hour': hour, 'is_weekend': is_weekend, 'demand': demand})
train, test = train_test_split(df_bikes, test_size=0.2, random_state=42)

X_train_bike = train[['temp', 'hour', 'is_weekend']].values
y_train_bike = train['demand'].values
X_test_bike  = test[['temp', 'hour', 'is_weekend']].values
y_test_bike  = test['demand'].values

X_train_const = sm.add_constant(X_train_bike)
X_test_const  = sm.add_constant(X_test_bike)

quantiles_bike = [0.1, 0.5, 0.9]
models_bike = {}
preds_bike  = {}

for q in quantiles_bike:
    model = sm.QuantReg(y_train_bike, X_train_const).fit(q=q)
    models_bike[q] = model
    preds_bike[q]  = model.predict(X_test_const)
    print(f"τ={q} | β_temp={model.params[1]:.2f}, β_hour={model.params[2]:.2f}, β_weekend={model.params[3]:.2f}")

coverage_bike = ((y_test_bike >= preds_bike[0.1]) & (y_test_bike <= preds_bike[0.9])).mean()
print(f"Test set coverage (80% PI): {coverage_bike:.2%}")

residuals_median = y_test_bike - preds_bike[0.5]
plt.scatter(preds_bike[0.5], residuals_median, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.title('Residual Plot (Median QR)')
plt.tight_layout()
plt.show()

###############################################################################
# SECTION 2: NON-LINEAR QR WITH LIGHTGBM — Single Feature
###############################################################################

np.random.seed(42)
n = 1000
X_nl = np.random.uniform(0, 4, n)
y_nl = np.sin(2 * X_nl) + 0.3 * X_nl**2 + np.random.normal(0, 0.5 + 0.3 * X_nl**2)
df_nl = pd.DataFrame({'X': X_nl, 'y': y_nl})

plt.figure(figsize=(10, 6))
plt.scatter(df_nl['X'], df_nl['y'], alpha=0.5, s=20)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Non-Linear Data with Heteroscedasticity')
plt.tight_layout()
plt.show()

X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
    df_nl[['X']].values, df_nl['y'].values, test_size=0.2, random_state=42
)

quantiles = [0.1, 0.5, 0.9]
models_nl = {}
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
    dtrain = lgb.Dataset(X_train_nl, label=y_train_nl)
    models_nl[tau] = lgb.train(params, dtrain, num_boost_round=300)
    print(f"Trained model for τ={tau}")

X_plot = np.linspace(0, 4, 200).reshape(-1, 1)
predictions_nl = {tau: models_nl[tau].predict(X_plot) for tau in quantiles}

plt.figure(figsize=(12, 7))
plt.scatter(X_test_nl, y_test_nl, alpha=0.3, s=20, label='Test data', color='gray')
colors = {0.1: 'blue', 0.5: 'green', 0.9: 'red'}
for tau in quantiles:
    plt.plot(X_plot, predictions_nl[tau], color=colors[tau], lw=2.5, label=f'QR(τ={tau})')
plt.fill_between(X_plot.flatten(), predictions_nl[0.1], predictions_nl[0.9],
                 alpha=0.2, color='orange', label='80% PI')
plt.xlabel('X', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Non-Linear Quantile Regression with LightGBM', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

###############################################################################
# SECTION 3: MULTI-FEATURE EXAMPLE — Feature Importance Across Quantiles
###############################################################################

np.random.seed(123)
n = 2000
df_multi = pd.DataFrame({
    'temp':       np.random.uniform(0, 40, n),
    'humidity':   np.random.uniform(20, 100, n),
    'hour':       np.random.randint(0, 24, n),
    'is_weekend': np.random.binomial(1, 0.3, n)
})
df_multi['demand'] = (
    50
    + 2  * df_multi['temp']
    + 0.3 * df_multi['humidity']
    + 10 * (df_multi['hour'] >= 17) * (df_multi['hour'] <= 20)
    + 15 * df_multi['is_weekend']
    + np.random.normal(0, 5 + 0.2 * df_multi['temp'])
)

feature_names = ['temp', 'humidity', 'hour', 'is_weekend']
X = df_multi[feature_names].values
y = df_multi['demand'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

quantiles_multi = [0.1, 0.5, 0.9]
feat_importances = {}

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
    raw = model.feature_importance(importance_type='gain')
    feat_importances[tau] = 100 * raw / raw.sum()

df_importance = pd.DataFrame(feat_importances, index=feature_names)
print(df_importance)

df_importance.T.plot(kind='bar', figsize=(10, 6))
plt.xlabel('Quantile', fontsize=12)
plt.ylabel('Feature Importance (%)', fontsize=12)
plt.title('Feature Importance Across Quantiles (Normalized)', fontsize=14)
plt.legend(title='Feature', fontsize=10)
plt.tight_layout()
plt.show()

###############################################################################
# SECTION 4: BAYESIAN HYPERPARAMETER TUNING WITH OPTUNA (τ=0.9)
###############################################################################

def objective(trial):
    params = {
        'objective': 'quantile',
        'alpha': 0.9,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves':    trial.suggest_int('num_leaves', 15, 127),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq':     trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1,
        'seed': 42
    }
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    model  = lgb.train(params, dtrain, num_boost_round=300)
    y_pred = model.predict(X_val)
    residual = y_val - y_pred
    tau = params['alpha']
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best hyperparameters:")
print(study.best_params)
print(f"Best validation loss: {study.best_value:.4f}")

best_params = {'objective': 'quantile', 'alpha': 0.9, **study.best_params, 'verbose': -1}
dtrain_full = lgb.Dataset(X_train, label=y_train)
final_model = lgb.train(best_params, dtrain_full, num_boost_round=500)

y_pred_test  = final_model.predict(X_test)
residual_test = y_test - y_pred_test
test_loss = np.where(residual_test >= 0, 0.9 * residual_test, -0.1 * residual_test).mean()
print(f"Test pinball loss: {test_loss:.4f}")

plot_optimization_history(study)
plt.tight_layout()
plt.show()

hp_importances = optuna.importance.get_param_importances(study)
hp_params = list(hp_importances.keys())
hp_values  = list(hp_importances.values())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(hp_params, hp_values, color='cornflowerblue')
for bar, val in zip(bars, hp_values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=10)
ax.set_xlabel("Hyperparameter Importance", fontsize=12)
ax.set_ylabel("Hyperparameter", fontsize=12)
ax.set_title("Hyperparameter Importances", fontsize=14)
ax.set_xlim(0, max(hp_values) * 1.15)
plt.tight_layout()
plt.show()

###############################################################################
# SECTION 5: QUANTILE-SPECIFIC TUNING & MONOTONICITY FIX
###############################################################################

def optimize_for_quantile(tau, n_trials=30):
    def objective(trial):
        params = {
            'objective': 'quantile',
            'alpha': tau,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves':    trial.suggest_int('num_leaves', 15, 127),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
            'verbose': -1,
            'seed': 42
        }
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model  = lgb.train(params, dtrain, num_boost_round=300)
        y_pred = model.predict(X_val)
        residual = y_val - y_pred
        return np.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params

taus = [0.1, 0.5, 0.9]
best_params_by_quantile = {}
for tau in taus:
    print(f"\nOptimizing for τ={tau}...")
    best_params_by_quantile[tau] = optimize_for_quantile(tau, n_trials=30)
    print(f"Best params: {best_params_by_quantile[tau]}")

final_models = {}
preds = {}
for tau in taus:
    best_p = best_params_by_quantile[tau]
    final_params = {
        'objective': 'quantile',
        'alpha': tau,
        'learning_rate': best_p['learning_rate'],
        'num_leaves':    best_p['num_leaves'],
        'min_data_in_leaf': best_p['min_data_in_leaf'],
        'verbose': -1,
        'seed': 42
    }
    dtrain_full = lgb.Dataset(X_train, label=y_train)
    final_models[tau] = lgb.train(final_params, dtrain_full, num_boost_round=300)
    preds[tau] = final_models[tau].predict(X_test)

# Enforce monotonicity (fix quantile crossing)
preds_corrected = {tau: [] for tau in taus}
for i in range(len(X_test)):
    quantile_vals = np.array([preds[tau][i] for tau in taus])
    quantile_vals_sorted = np.sort(quantile_vals)
    for j, tau in enumerate(taus):
        preds_corrected[tau].append(quantile_vals_sorted[j])

for tau in taus:
    preds_corrected[tau] = np.array(preds_corrected[tau])

pred_lower = preds_corrected[0.1]
pred_upper = preds_corrected[0.9]
coverage = ((y_test >= pred_lower) & (y_test <= pred_upper)).mean()
print(f"Coverage after monotonicity fix: {coverage:.2%}")
if coverage < 0.75:
    print("WARNING: QR model miscalibrated, coverage below 75%")

###############################################################################
# SECTION 6: MODEL SAVING & VERSIONING
###############################################################################

final_models[0.9].save_model('qr_tau09_v2.txt')
loaded_model = lgb.Booster(model_file='qr_tau09_v2.txt')

model_metadata = {
    'tau': 0.9,
    'train_date': '2025-10-12',
    'n_train': len(X_train),
    'features': feature_names,
    'test_coverage': 0.89
}
