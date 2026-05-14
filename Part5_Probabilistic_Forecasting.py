# -*- coding: utf-8 -*-
"""
Part 5 — The State of the Art: Probabilistic Forecasting with Quantile Regression
Dense quantile grids, QRF, Deep QR, calibration, conformal prediction, and deployment.
Author: 2577
"""

###############################################################################
# IMPORTS
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import lightgbm as lgb
import optuna

plt.style.use('seaborn')

###############################################################################
# DATA — Non-Linear with Heteroscedasticity (shared across all sections)
###############################################################################

np.random.seed(42)
n = 2000
X_raw = np.random.uniform(0, 4, n)
y_raw = np.sin(2 * X_raw) + 0.3 * X_raw**2 + np.random.normal(0, 0.5 + 0.3 * X_raw**2)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw.reshape(-1, 1), y_raw, test_size=0.2, random_state=42
)

###############################################################################
# SECTION 1: DENSE QUANTILE GRID — Full CDF Reconstruction
###############################################################################

taus = np.round(np.arange(0.01, 1.00, 0.01), 2)  # 99 quantiles

def fit_quantile_model(tau, X_train, y_train):
    params = {
        'objective': 'quantile',
        'alpha': tau,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 30,
        'verbose': -1,
        'seed': 42
    }
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=300)
    return tau, model

# Parallel training (adjust n_jobs to your CPU cores)
results = Parallel(n_jobs=4)(
    delayed(fit_quantile_model)(tau, X_train, y_train) for tau in taus
)
models = dict(results)
print(f"Trained {len(models)} quantile models")

# Predict full distribution for a single test point
x_test_point = np.array([[2.5]])
quantiles_pred = {tau: models[tau].predict(x_test_point)[0] for tau in taus}

plt.figure(figsize=(10, 6))
plt.plot(list(quantiles_pred.values()), taus, lw=2.5, color='steelblue')
plt.xlabel('y (predicted)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.title('Predicted CDF at X=2.5', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Probabilistic queries
taus_list      = list(taus)
quantiles_list = list(quantiles_pred.values())

tau_at_5      = np.interp(5, quantiles_list, taus_list)
prob_exceed_5 = 1 - tau_at_5
expected_y    = np.mean(quantiles_list)
q_05 = quantiles_pred[0.05]
q_95 = quantiles_pred[0.95]

print(f"P(Y > 5 | X=2.5)  ≈ {prob_exceed_5:.2%}")
print(f"E[Y | X=2.5]       ≈ {expected_y:.3f}")
print(f"90% PI: [{q_05:.3f}, {q_95:.3f}]")

# Save models and test data for calibration section
saved_test_data = {'X_test': X_test.copy(), 'y_test': y_test.copy(), 'models': models}
with open('lgb_models.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('saved_test_data.pkl', 'wb') as f:
    pickle.dump(saved_test_data, f)
print("Saved LightGBM models and test data to disk")

###############################################################################
# SECTION 2: QUANTILE REGRESSION FORESTS (QRF)
###############################################################################

try:
    from quantile_forest import RandomForestQuantileRegressor

    qrf = RandomForestQuantileRegressor(n_estimators=100, min_samples_leaf=10, random_state=42)
    qrf.fit(X_train, y_train)

    taus_qrf    = np.arange(0.01, 1.00, 0.01)
    y_pred_qrf  = qrf.predict(X_test, quantiles=taus_qrf.tolist())
    print(f"QRF: predicted {len(taus_qrf)} quantiles for {len(X_test)} test points")

    test_idx        = 0
    quantiles_qrf   = y_pred_qrf[test_idx, :]

    plt.figure(figsize=(10, 6))
    plt.plot(quantiles_qrf, taus_qrf, lw=2.5, color='darkgreen', label='QRF CDF')
    plt.xlabel('y (predicted)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title(f'QRF: Predicted CDF at X={X_test[test_idx][0]:.2f}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

except ImportError:
    print("quantile-forest not installed. Run: pip install quantile-forest")

###############################################################################
# SECTION 3: DEEP QUANTILE REGRESSION (PyTorch)
###############################################################################

class DeepQuantileRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, quantiles):
        super().__init__()
        self.quantiles = quantiles
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in quantiles])

    def forward(self, x):
        h = self.shared(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


class DeepQuantileRegressorMonotonic(nn.Module):
    """Variant with cumsum trick to prevent quantile crossing."""
    def __init__(self, input_dim, hidden_dim, quantiles):
        super().__init__()
        self.quantiles = sorted(quantiles)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Linear(hidden_dim, len(quantiles))

    def forward(self, x):
        h = self.shared(x)
        increments = torch.exp(self.head(h))
        return torch.cumsum(increments, dim=1)


def pinball_loss_torch(y_true, y_pred, tau):
    residual = y_true - y_pred
    return torch.where(residual >= 0, tau * residual, (tau - 1) * residual).mean()


# Fresh data for PyTorch (ensures contiguous memory layout)
np.random.seed(42)
n_torch = 2000
X_fresh = np.random.uniform(0, 4, n_torch)
y_fresh = np.sin(2 * X_fresh) + 0.3 * X_fresh**2 + np.random.normal(0, 0.5 + 0.3 * X_fresh**2)

X_train_torch, X_test_torch, y_train_torch, y_test_torch = train_test_split(
    X_fresh.reshape(-1, 1), y_fresh, test_size=0.2, random_state=42
)

X_tensor = torch.FloatTensor(np.ascontiguousarray(X_train_torch))
y_tensor = torch.FloatTensor(np.ascontiguousarray(y_train_torch)).reshape(-1, 1)
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True, num_workers=0)

quantiles_deep = [0.1, 0.5, 0.9]
deep_model = DeepQuantileRegressor(input_dim=1, hidden_dim=128, quantiles=quantiles_deep)
optimizer  = torch.optim.Adam(deep_model.parameters(), lr=0.001)

for epoch in tqdm(range(100)):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = deep_model(X_batch)
        loss = sum(pinball_loss_torch(y_batch, y_pred[:, i:i+1], tau)
                   for i, tau in enumerate(quantiles_deep))
        loss.backward()
        optimizer.step()

print("Deep QR training complete")

###############################################################################
# SECTION 4: BAYESIAN TUNING FOR DEEP QR (Optuna)
###############################################################################

def deep_qr_objective(trial):
    hidden_dim    = trial.suggest_int('hidden_dim', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout       = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size    = trial.suggest_categorical('batch_size', [32, 64, 128])

    model_t = DeepQuantileRegressor(input_dim=1, hidden_dim=hidden_dim, quantiles=[0.1, 0.5, 0.9])
    for layer in model_t.shared:
        if isinstance(layer, nn.Dropout):
            layer.p = dropout
    opt_t = torch.optim.Adam(model_t.parameters(), lr=learning_rate)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=42)
    loader_t = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).reshape(-1, 1)),
        batch_size=batch_size, shuffle=True
    )

    model_t.train()
    for _ in range(50):
        for X_b, y_b in loader_t:
            opt_t.zero_grad()
            y_p = model_t(X_b)
            loss = sum(pinball_loss_torch(y_b, y_p[:, i:i+1], tau)
                       for i, tau in enumerate([0.1, 0.5, 0.9]))
            loss.backward()
            opt_t.step()

    model_t.eval()
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
    with torch.no_grad():
        y_p_val = model_t(X_val_t)
        val_loss = sum(pinball_loss_torch(y_val_t, y_p_val[:, i:i+1], tau).item()
                       for i, tau in enumerate([0.1, 0.5, 0.9]))
    return val_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_deep = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_deep.optimize(deep_qr_objective, n_trials=30, show_progress_bar=True)

print("Best deep QR hyperparameters:")
print(study_deep.best_params)
print(f"Best validation loss: {study_deep.best_value:.4f}")

# Retrain with best params
best = study_deep.best_params
final_deep_model = DeepQuantileRegressor(input_dim=1, hidden_dim=best['hidden_dim'], quantiles=[0.1, 0.5, 0.9])
for layer in final_deep_model.shared:
    if isinstance(layer, nn.Dropout):
        layer.p = best['dropout']

opt_final = torch.optim.Adam(final_deep_model.parameters(), lr=best['learning_rate'])
full_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_torch), torch.FloatTensor(y_train_torch).reshape(-1, 1)),
    batch_size=best['batch_size'], shuffle=True
)

for epoch in range(100):
    final_deep_model.train()
    for X_b, y_b in full_loader:
        opt_final.zero_grad()
        y_p = final_deep_model(X_b)
        loss = sum(pinball_loss_torch(y_b, y_p[:, i:i+1], tau)
                   for i, tau in enumerate([0.1, 0.5, 0.9]))
        loss.backward()
        opt_final.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1} complete")

print("Final deep QR model trained with best hyperparameters")

###############################################################################
# SECTION 5: CALIBRATION
###############################################################################

# Load saved LightGBM models and test data
with open('saved_test_data.pkl', 'rb') as f:
    saved = pickle.load(f)
X_test_cal = saved['X_test']
y_test_cal = saved['y_test']
models_cal = saved['models']
taus_cal   = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Predictions for all test points
quantile_preds_test = {tau: models_cal[tau].predict(X_test_cal) for tau in taus_cal}


def quantile_calibration_plot(y_test, quantile_preds, taus):
    empirical = [(y_test <= quantile_preds[tau]).mean() for tau in taus]
    plt.figure(figsize=(7, 7))
    plt.plot(taus, empirical, 'o-', lw=2, markersize=4, label='Empirical')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
    plt.xlabel('Nominal Quantile (τ)', fontsize=12)
    plt.ylabel('Empirical Quantile', fontsize=12)
    plt.title('Quantile Calibration Plot', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

quantile_calibration_plot(y_test_cal, quantile_preds_test, taus_cal)

# Coverage
lower_cal = quantile_preds_test[0.1]
upper_cal  = quantile_preds_test[0.9]
coverage   = ((y_test_cal >= lower_cal) & (y_test_cal <= upper_cal)).mean()
print(f"80% interval coverage: {coverage:.2%} (target: 80%)")


def winkler_score(y_true, lower, upper, alpha=0.2):
    width         = upper - lower
    penalty_lower = (2 / alpha) * np.maximum(0, lower - y_true)
    penalty_upper = (2 / alpha) * np.maximum(0, y_true - upper)
    avg_width     = width.mean()
    avg_penalty   = (penalty_lower + penalty_upper).mean()
    coverage      = ((y_true >= lower) & (y_true <= upper)).mean()
    total_score   = (width + penalty_lower + penalty_upper).mean()
    penalty_frac  = avg_penalty / total_score if total_score > 0 else 0
    print(f"Winkler Score Components:")
    print(f"  Average width:   {avg_width:.4f}")
    print(f"  Average penalty: {avg_penalty:.4f} ({100*penalty_frac:.1f}% of total)")
    print(f"  Coverage:        {coverage:.2%} (target: {100*(1-alpha):.0f}%)")
    return total_score

ws = winkler_score(y_test_cal, quantile_preds_test[0.1], quantile_preds_test[0.9])
print(f"Winkler score: {ws:.4f}")


def crps_quantile(y_true, quantile_preds_array, taus):
    n = len(y_true)
    crps_sum = 0
    for i in range(n):
        y  = y_true[i]
        qs = quantile_preds_array[i]
        score = 0
        for k, tau in enumerate(taus):
            residual = y - qs[k]
            score += tau * residual if residual >= 0 else (tau - 1) * residual
        crps_sum += score / len(taus)
    crps_val = crps_sum / n
    print(f"CRPS: {crps_val:.4f}  |  CRPS/y_std: {crps_val/y_true.std():.4f}")
    return crps_val

quantile_preds_array = np.array([quantile_preds_test[tau] for tau in taus_cal]).T
crps_quantile(y_test_cal, quantile_preds_array, taus_cal)

###############################################################################
# SECTION 6: CONFORMAL PREDICTION
###############################################################################

X_tr_cf, X_cal_cf, y_tr_cf, y_cal_cf = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model_lower_cf = lgb.train({'objective': 'quantile', 'alpha': 0.05, 'verbose': -1},
                            lgb.Dataset(X_tr_cf, label=y_tr_cf), num_boost_round=300)
model_upper_cf = lgb.train({'objective': 'quantile', 'alpha': 0.95, 'verbose': -1},
                            lgb.Dataset(X_tr_cf, label=y_tr_cf), num_boost_round=300)

pred_lower_cal_cf = model_lower_cf.predict(X_cal_cf)
pred_upper_cal_cf = model_upper_cf.predict(X_cal_cf)

residuals = np.maximum(y_cal_cf - pred_lower_cal_cf, pred_upper_cal_cf - y_cal_cf)
alpha_cf  = 0.1  # Target: 90% coverage
q_adjust  = np.percentile(residuals, (1 - alpha_cf) * 100)

pred_lower_test_cf = model_lower_cf.predict(X_test) - q_adjust
pred_upper_test_cf = model_upper_cf.predict(X_test) + q_adjust

coverage_cf = ((y_test >= pred_lower_test_cf) & (y_test <= pred_upper_test_cf)).mean()
print(f"Conformal prediction coverage: {coverage_cf:.2%} (target: 90%)")

###############################################################################
# SECTION 7: PRODUCTION MONITORING
###############################################################################

def monitor_coverage(y_actual, preds_lower, preds_upper, target_coverage=0.8):
    empirical_coverage = ((y_actual >= preds_lower) & (y_actual <= preds_upper)).mean()
    print(f"Coverage: {empirical_coverage:.2%} (target: {target_coverage:.0%})")
    if empirical_coverage < target_coverage - 0.05:
        print("WARNING: Coverage dropped — consider retraining the model")
    return empirical_coverage

monitor_coverage(y_test_cal, quantile_preds_test[0.1], quantile_preds_test[0.9])
