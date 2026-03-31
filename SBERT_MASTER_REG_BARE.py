"""
SBERT-based Mass Regression Model with MLP Head

Trains a neural network regressor using sentence embeddings (SBERT) as features
to predict component mass. Includes K-fold cross-validation with proper data
separation, multiple baseline comparisons, and optional Optuna hyperparameter
tuning.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import optuna
from optuna.samplers import TPESampler


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Data paths ---
    'train_csv': 'CSM - Master - EC.csv',
    # Set to 'NONE' (string) or None to skip prediction data entirely.
    # When set to NONE, no duplicate removal between train and predict is done.
    'predict_csv': 'NONE',
    'text_col': 'Name',
    'target_col': 'Mass (g)',
    'drop_unnamed_columns': True,

    # --- Outlier filtering ---
    # Method: None (no filtering), 'iqr', 'percentile', or 'mass_cutoff'
    'outlier_method': 'percentile',
    # IQR settings (used when outlier_method == 'iqr')
    'iqr_multiplier': 15.0,
    # Percentile settings (used when outlier_method == 'percentile')
    'percentile_lower': 0.0,
    'percentile_upper': 95,
    # Mass cutoff settings (used when outlier_method == 'mass_cutoff')
    # Removes any datapoint with mass >= this value (in grams)
    'mass_cutoff_g': 25.0,

    # --- Embedding model ---
    'embedding_model': 'all-minilm-l12-v2',

    # --- Cross-validation ---
    'cv_folds': 10,
    'random_seed': 42,

    # --- Neural network hyperparameters ---
    'batch_size': 128,
    'epochs': 200,
    'lr': 3e-4,
    'hidden_dim': 256,
    'dropout': 0.1,
    'early_stopping_patience': 40,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Loss function: 'mse', 'mae', or 'huber'
    'loss_function': 'huber',
    'huber_delta': 0.5,

    # Target transformation: None or 'log'
    'target_normalization': None,

    # --- Optuna hyperparameter tuning ---
    'use_optuna': False,
    'optuna_n_trials': 50,
    'optuna_timeout_s': None,

    # --- Output paths ---
    'save_r2_summary_csv': 'SBERT_ASP3_Acc_Metrics_NEWEST_EC.csv',
    'save_pred_stats_csv': 'SBERT_ASP3_Predictions_NEWEST_EC.csv',
    'save_baseline_values_csv': 'SBERT_ASP3_Baseline_Values_NEWEST_EC.csv',
    'save_dataset_summary_csv': 'SBERT_ASP3_Dataset_Summary_NEWEST_EC.csv',
    'save_dataset_by_category_csv': 'SBERT_ASP3_Dataset_By_Category_NEWEST_EC.csv',
    'save_plots_dir': 'SBERT_ASP3_Plots_NEWEST_EC',

    'verbose': True,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# METRICS
# =============================================================================
def r2(y, p):
    """Calculate R² score, returning NaN on failure."""
    try:
        return r2_score(y, p)
    except Exception:
        return np.nan


def mae(y, p):
    """Calculate Mean Absolute Error, returning NaN on failure."""
    try:
        return mean_absolute_error(y, p)
    except Exception:
        return np.nan


def rmse(y, p):
    """Calculate Root Mean Squared Error, returning NaN on failure."""
    try:
        return math.sqrt(mean_squared_error(y, p))
    except Exception:
        return np.nan


def summarise_metric(values):
    """Compute mean and sample standard deviation for a list of fold metrics."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return {'mean': np.nanmean(arr) if len(arr) > 0 else np.nan, 'std': np.nan}
    return {'mean': np.mean(arr), 'std': np.std(arr, ddof=1)}


# =============================================================================
# PYTORCH DATASET
# =============================================================================
class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embedding vectors with optional targets."""

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# =============================================================================
# MODEL
# =============================================================================
class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron for regression with positive-only output.

    Architecture:
        Input -> Hidden   (LeakyReLU, Dropout, BatchNorm)
        Hidden -> Hidden/2 (LeakyReLU, Dropout, BatchNorm)
        Hidden/2 -> 1      (Softplus for strictly positive output)
    """

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),

            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def get_loss_fn(name, huber_delta=1.0):
    """
    Get loss function by name.

    Parameters
    ----------
    name : str
        One of 'mse', 'mae', 'huber'
    huber_delta : float
        Threshold for Huber loss (smaller = more robust to outliers)
    """
    if name == 'mse':
        return nn.MSELoss()
    if name == 'mae':
        return nn.L1Loss()
    if name == 'huber':
        return nn.SmoothL1Loss(beta=huber_delta)
    return nn.MSELoss()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, optimizer, device, loss_fn):
    """Train for one epoch, return mean loss."""
    model.train()
    losses = []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses) if losses else np.nan


def eval_epoch(model, loader, device):
    """Evaluate model, return predictions and true values."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds.extend(model(Xb).cpu().numpy())
            trues.extend(yb.cpu().numpy())
    return np.array(preds), np.array(trues)


def predict(model, loader, device):
    """Generate predictions for a dataset (no labels required)."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            Xb = batch[0] if isinstance(batch, (tuple, list)) else batch
            out.extend(model(Xb.to(device)).cpu().numpy())
    return np.array(out)


def train_nn(cfg, X_train, y_train, X_val, y_val, params_override=None):
    """
    Train neural network with early stopping.

    Returns
    -------
    model : MLPRegressor
        Trained model (best checkpoint).
    inv_transform : callable
        Inverse transform for target (identity or expm1).
    history : dict
        Training history with 'train_loss', 'val_r2', 'val_mae' per epoch.
    """
    params = {
        'lr': cfg['lr'],
        'hidden_dim': cfg['hidden_dim'],
        'dropout': cfg['dropout'],
        'epochs': cfg['epochs'],
        'batch_size': cfg['batch_size'],
        'early_stopping_patience': cfg['early_stopping_patience'],
        'loss_function': cfg['loss_function'],
        'huber_delta': cfg.get('huber_delta', 1.0),
    }
    if params_override:
        params.update(params_override)

    # Optional log transform of target
    if cfg.get('target_normalization') == 'log':
        ytr, yv = np.log1p(y_train), np.log1p(y_val)
        inv = np.expm1
    else:
        ytr, yv = y_train.copy(), y_val.copy()
        inv = lambda x: x

    train_loader = DataLoader(
        EmbeddingDataset(X_train, ytr),
        batch_size=params['batch_size'], shuffle=True,
    )
    val_loader = DataLoader(
        EmbeddingDataset(X_val, yv),
        batch_size=params['batch_size'], shuffle=False,
    )

    model = MLPRegressor(
        X_train.shape[1],
        hidden_dim=params['hidden_dim'],
        dropout=params['dropout'],
    ).to(cfg['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    loss_fn = get_loss_fn(params['loss_function'], huber_delta=params['huber_delta'])

    best_state, best_r2 = None, -np.inf
    no_imp = 0
    history = {'train_loss': [], 'val_r2': [], 'val_mae': []}

    for epoch in tqdm(range(params['epochs']), desc='NN epochs', leave=False):
        train_loss = train_epoch(model, train_loader, optimizer, cfg['device'], loss_fn)
        p_norm, t_norm = eval_epoch(model, val_loader, cfg['device'])

        p = inv(p_norm) if cfg.get('target_normalization') == 'log' else p_norm
        t = inv(t_norm) if cfg.get('target_normalization') == 'log' else t_norm

        val_r2 = r2(t, p)
        val_mae = mae(t, p)

        history['train_loss'].append(train_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)

        if np.isfinite(val_r2) and val_r2 > best_r2:
            best_r2 = val_r2
            best_state = model.state_dict()
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= params['early_stopping_patience']:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, inv, history


# =============================================================================
# PLOTTING
# =============================================================================
def setup_plot_style():
    """Configure matplotlib for publication-quality plots (Times New Roman)."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 1500,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 16,
        'axes.labelsize': 19,
        'axes.titlesize': 21,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
    })


def save_learning_curve(history, save_dir, prefix='NN'):
    """Save learning curve plot (training loss, validation R², validation MAE)."""
    setup_plot_style()
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Training Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Validation R²
    axes[1].plot(epochs, history['val_r2'], 'g-', linewidth=2, label='Validation R²')
    axes[1].axhline(y=max(history['val_r2']), color='r', linestyle='--', alpha=0.5,
                     label=f'Best: {max(history["val_r2"]):.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Validation R²')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Validation MAE
    axes[2].plot(epochs, history['val_mae'], 'orange', linewidth=2, label='Validation MAE')
    axes[2].axhline(y=min(history['val_mae']), color='r', linestyle='--', alpha=0.5,
                     label=f'Best: {min(history["val_mae"]):.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Validation MAE')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_learning_curve.svg"),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved learning curve plot")


def save_hexbin_plot(y_true, y_pred, save_dir, prefix, target_col='Mass (g)'):
    """
    Save combined hexbin figure: panel (a) reported vs predicted,
    panel (b) residuals vs reported, with a shared log-scale colorbar.
    """
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    setup_plot_style()

    CMAP = 'inferno'
    REF_COLOR = '#D3D3D3'
    REF_ALPHA = 0.6

    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if valid.sum() == 0:
        print(f"Warning: No valid predictions for {prefix}, skipping hexbin plot")
        return

    yt, yp = y_true[valid], y_pred[valid]
    resid = yt - yp

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Probe colour range from the predicted-vs-reported panel
    h_temp = ax1.hexbin(yt, yp, gridsize=50, cmap=CMAP, mincnt=1,
                        edgecolors='none', norm=LogNorm())
    vmin, vmax = h_temp.get_clim()
    ax1.cla()
    shared_norm = LogNorm(vmin=vmin, vmax=vmax)

    # Panel (a): Reported vs Predicted
    ax1.hexbin(yt, yp, gridsize=50, cmap=CMAP, mincnt=1,
               edgecolors='none', norm=shared_norm)
    mn = min(yt.min(), yp.min())
    mx = max(yt.max(), yp.max())
    ax1.plot([mn, mx], [mn, mx], color=REF_COLOR, linewidth=2,
             linestyle='--', alpha=REF_ALPHA)
    ax1.set_xlabel(f'Reported {target_col}')
    ax1.set_ylabel(f'Predicted {target_col}')
    ax1.set_title('(a)')
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)

    # Panel (b): Residuals vs Reported
    h2 = ax2.hexbin(yt, resid, gridsize=50, cmap=CMAP, mincnt=1,
                    edgecolors='none', norm=shared_norm)
    ax2.axhline(0, color=REF_COLOR, linestyle='--', linewidth=2, alpha=REF_ALPHA)
    ax2.set_xlabel(f'Reported {target_col}')
    ax2.set_ylabel('Residual (Reported − Predicted)')
    ax2.set_title('(b)')
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)

    # Shared colorbar
    cb = fig.colorbar(h2, cax=cbar_ax)
    cb.set_label('Count (log scale)')

    save_path = os.path.join(save_dir, f"{prefix}_hexbin_combined.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved combined hexbin plot")


def save_residual_histogram(y_true, y_pred, save_dir, prefix):
    """Save residual histogram with log y-scale."""
    setup_plot_style()

    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if valid.sum() == 0:
        print(f"Warning: No valid predictions for {prefix}, skipping residual histogram")
        return

    resid = y_true[valid] - y_pred[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(resid, bins=50, color='#4878CF', edgecolor='#3a5fa8', linewidth=0.4,
            alpha=0.92, zorder=2)
    ax.axvline(0, color='#222222', linestyle='-', linewidth=1.5, alpha=0.7, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Residual (Reported − Predicted)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_residual_histogram.svg"),
                format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved residual histogram")


# =============================================================================
# PREDICTION AGGREGATION
# =============================================================================
def aggregate_predictions(pred_list):
    """Aggregate predictions across folds, returning mean and std."""
    valid = [p for p in pred_list if p is not None]
    if len(valid) == 0:
        return None, None
    mat = np.vstack(valid)
    return mat.mean(axis=0), mat.std(axis=0)


# =============================================================================
# DATASET SUMMARY FUNCTIONS
# =============================================================================
def compute_dataset_summary(df_before_outlier, df_after_outlier, df_final, cfg):
    """
    Compute overall dataset summary statistics at each cleaning stage.

    Returns
    -------
    pd.DataFrame with summary statistics.
    """
    target = cfg['target_col']
    text_col = cfg['text_col']

    stages = [
        ('Before_Outlier', df_before_outlier),
        ('After_Outlier', df_after_outlier),
        ('Final', df_final),
    ]

    rows = []

    # Dataset size metrics
    rows.append({
        'Metric': 'Total Rows (after missing removal)',
        **{label: len(d) for label, d in stages},
    })
    rows.append({
        'Metric': 'Rows Removed by Outlier Filter',
        'Before_Outlier': '-',
        'After_Outlier': len(df_before_outlier) - len(df_after_outlier),
        'Final': '-',
    })
    rows.append({
        'Metric': 'Rows Removed by Overlap Filter',
        'Before_Outlier': '-',
        'After_Outlier': '-',
        'Final': len(df_after_outlier) - len(df_final),
    })
    rows.append({
        'Metric': 'Unique Component Categories',
        **{label: d['Component Category'].nunique() for label, d in stages},
    })

    # Target variable statistics
    stat_fns = [
        ('Mass - Min', lambda s: s.min()),
        ('Mass - Max', lambda s: s.max()),
        ('Mass - Range', lambda s: s.max() - s.min()),
        ('Mass - Mean', lambda s: s.mean()),
        ('Mass - Median', lambda s: s.median()),
        ('Mass - Std Dev', lambda s: s.std()),
        ('Mass - Q1 (25%)', lambda s: s.quantile(0.25)),
        ('Mass - Q3 (75%)', lambda s: s.quantile(0.75)),
        ('Mass - IQR', lambda s: s.quantile(0.75) - s.quantile(0.25)),
        ('Mass - Skewness', lambda s: s.skew()),
        ('Mass - Kurtosis', lambda s: s.kurtosis()),
    ]
    for metric_name, fn in stat_fns:
        rows.append({
            'Metric': metric_name,
            **{label: fn(d[target]) for label, d in stages},
        })

    # Text feature statistics (final dataset only)
    text_lengths = df_final[text_col].astype(str).str.len()
    word_counts = df_final[text_col].astype(str).str.split().str.len()
    all_words = ' '.join(df_final[text_col].astype(str)).lower().split()
    vocab_size = len(set(all_words))

    for metric_name, value in [
        ('Text - Mean Length (chars)', text_lengths.mean()),
        ('Text - Median Length (chars)', text_lengths.median()),
        ('Text - Mean Word Count', word_counts.mean()),
        ('Text - Median Word Count', word_counts.median()),
        ('Text - Vocabulary Size (unique words)', vocab_size),
    ]:
        rows.append({
            'Metric': metric_name,
            'Before_Outlier': '-', 'After_Outlier': '-', 'Final': value,
        })

    return pd.DataFrame(rows)


def compute_category_stats(df, cfg):
    """
    Compute per-category statistics for the dataset.

    Returns
    -------
    pd.DataFrame with per-category statistics.
    """
    target = cfg['target_col']
    grouped = df.groupby('Component Category')[target]

    def _stats(cat_data, cat_name, total_n):
        return {
            'Component_Category': cat_name,
            'Count': len(cat_data),
            'Percent_of_Total': 100 * len(cat_data) / total_n,
            'Mass_Min': cat_data.min(),
            'Mass_Max': cat_data.max(),
            'Mass_Range': cat_data.max() - cat_data.min(),
            'Mass_Mean': cat_data.mean(),
            'Mass_Median': cat_data.median(),
            'Mass_Std': cat_data.std(),
            'Mass_Q1': cat_data.quantile(0.25),
            'Mass_Q3': cat_data.quantile(0.75),
            'Mass_IQR': cat_data.quantile(0.75) - cat_data.quantile(0.25),
            'Mass_Skewness': cat_data.skew() if len(cat_data) > 2 else np.nan,
            'Mass_Kurtosis': cat_data.kurtosis() if len(cat_data) > 3 else np.nan,
        }

    total_n = len(df)
    rows = [_stats(grouped.get_group(cat), cat, total_n)
            for cat in sorted(df['Component Category'].unique())]
    rows.append(_stats(df[target], '*** TOTAL ***', total_n))

    result = pd.DataFrame(rows)
    total_row = result[result['Component_Category'] == '*** TOTAL ***']
    other_rows = result[result['Component_Category'] != '*** TOTAL ***'].sort_values(
        'Count', ascending=False)
    return pd.concat([other_rows, total_row], ignore_index=True)


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================
def optuna_objective(trial, cfg, X, y, df):
    """
    Optuna objective function for hyperparameter tuning.
    Uses 5-fold CV internally for faster evaluation.
    """
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        'epochs': cfg['epochs'],
        'early_stopping_patience': cfg['early_stopping_patience'],
    }

    kf_tune = KFold(n_splits=5, shuffle=True, random_state=cfg.get('random_seed', 42))
    fold_r2s = []

    for tune_fold, (tr_idx, val_idx) in enumerate(kf_tune.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)
        X_val = scaler.transform(X_val)

        try:
            model, inv_transform, _ = train_nn(
                cfg, X_tr, y_tr, X_val, y_val, params_override=params)
            val_loader = DataLoader(EmbeddingDataset(X_val), batch_size=params['batch_size'])
            preds_raw = predict(model, val_loader, cfg['device'])
            preds = inv_transform(preds_raw) if cfg.get('target_normalization') == 'log' else preds_raw
            fold_r2_val = r2(y_val, preds)
            if not np.isnan(fold_r2_val):
                fold_r2s.append(fold_r2_val)
            trial.report(fold_r2_val, tune_fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as e:
            if cfg.get('verbose', False):
                print(f"  Trial {trial.number} fold {tune_fold} failed: {e}")
            continue

    return np.mean(fold_r2s) if fold_r2s else float('-inf')


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================
def has_predict_csv(cfg):
    """Check whether a valid prediction CSV path is configured."""
    val = cfg.get('predict_csv')
    if val is None:
        return False
    if isinstance(val, str) and val.strip().upper() == 'NONE':
        return False
    return True


def load_and_clean_data(cfg):
    """
    Load training CSV, drop missing values, apply outlier filtering,
    optionally load prediction CSV and remove overlapping texts.

    Returns
    -------
    df : pd.DataFrame
        Final cleaned training data.
    df_pred : pd.DataFrame or None
        Prediction set (if provided).
    df_before_outlier : pd.DataFrame
        Snapshot before outlier removal.
    df_after_outlier : pd.DataFrame
        Snapshot after outlier removal.
    """
    print("Loading dataset...")
    df = pd.read_csv(cfg['train_csv'])

    if cfg.get('drop_unnamed_columns', True):
        unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
        if unnamed:
            print(f"Dropping unnamed columns: {unnamed}")
            df = df.drop(columns=unnamed)

    df = df.dropna(subset=[cfg['text_col'], cfg['target_col']]).reset_index(drop=True)
    print(f"After dropping missing values: {len(df)} rows")

    df_before_outlier = df.copy()

    # --- Outlier filtering ---
    outlier_method = cfg.get('outlier_method')
    target = cfg['target_col']

    if outlier_method == 'iqr':
        before = len(df)
        q1, q3 = df[target].quantile(0.25), df[target].quantile(0.75)
        iqr_val = q3 - q1
        mult = cfg.get('iqr_multiplier', 1.5)
        lower, upper = q1 - mult * iqr_val, q3 + mult * iqr_val
        df = df[(df[target] >= lower) & (df[target] <= upper)].reset_index(drop=True)
        print(f"IQR filtering (multiplier={mult}): removed {before - len(df)} rows; "
              f"{len(df)} remain")
        if len(df) > 0:
            print(f"  Mass range: [{df[target].min():.4f}, {df[target].max():.4f}]")

    elif outlier_method == 'percentile':
        before = len(df)
        lower_p = float(cfg.get('percentile_lower', 1.0))
        upper_p = float(cfg.get('percentile_upper', 99.0))
        if not (0 <= lower_p < upper_p <= 100):
            print(f"Invalid percentiles ({lower_p}, {upper_p}), skipping filtering.")
        else:
            lower = np.percentile(df[target], lower_p)
            upper = np.percentile(df[target], upper_p)
            df = df[(df[target] >= lower) & (df[target] <= upper)].reset_index(drop=True)
            print(f"Percentile filtering ({lower_p}th–{upper_p}th): removed "
                  f"{before - len(df)} rows; {len(df)} remain")
            if len(df) > 0:
                print(f"  Mass range: [{df[target].min():.4f}, {df[target].max():.4f}]")

    elif outlier_method == 'mass_cutoff':
        before = len(df)
        cutoff = float(cfg.get('mass_cutoff_g', 25.0))
        df = df[df[target] <= cutoff].reset_index(drop=True)
        print(f"Mass cutoff filtering (≤{cutoff} g): removed {before - len(df)} rows; "
              f"{len(df)} remain")
        if len(df) > 0:
            print(f"  Mass range: [{df[target].min():.4f}, {df[target].max():.4f}]")

    df_after_outlier = df.copy()

    # --- Load prediction set (optional) ---
    df_pred = None
    if has_predict_csv(cfg):
        try:
            df_pred = pd.read_csv(cfg['predict_csv'])
            df_pred = df_pred.dropna(subset=[cfg['text_col']]).reset_index(drop=True)
            # Remove overlapping texts from training set
            common = (set(df[cfg['text_col']].astype(str))
                      & set(df_pred[cfg['text_col']].astype(str)))
            if common:
                print(f"Removing {len(common)} overlapping texts from training set")
            df = df[~df[cfg['text_col']].astype(str).isin(
                df_pred[cfg['text_col']].astype(str))].reset_index(drop=True)
        except Exception as e:
            print(f"Could not load predict CSV: {e}")
            df_pred = None
    else:
        print("No prediction CSV configured — skipping prediction data loading.")

    return df, df_pred, df_before_outlier, df_after_outlier


# =============================================================================
# MAIN
# =============================================================================
def main(cfg=CONFIG):
    cfg = cfg.copy()

    # Reproducibility
    set_seed(cfg.get('random_seed', 42))
    print(f"Random seed set to: {cfg.get('random_seed', 42)}")

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    df, df_pred, df_before_outlier, df_after_outlier = load_and_clean_data(cfg)
    df_final = df.copy()

    # ------------------------------------------------------------------
    # 2. Compute and save dataset summaries
    # ------------------------------------------------------------------
    os.makedirs(cfg.get('save_plots_dir', 'plots'), exist_ok=True)

    print("\nGenerating dataset summary statistics...")
    df_dataset_summary = compute_dataset_summary(
        df_before_outlier, df_after_outlier, df_final, cfg)
    df_dataset_summary.to_csv(cfg['save_dataset_summary_csv'], index=False)
    print(f"Saved dataset summary to {cfg['save_dataset_summary_csv']}")

    df_dataset_by_category = compute_category_stats(df_final, cfg)
    df_dataset_by_category.to_csv(cfg['save_dataset_by_category_csv'], index=False)
    print(f"Saved dataset by category to {cfg['save_dataset_by_category_csv']}")

    # ------------------------------------------------------------------
    # 3. Compute and save baseline values
    # ------------------------------------------------------------------
    print("\nComputing baseline values from cleaned dataset...")
    global_mean = df[cfg['target_col']].mean()
    global_median = df[cfg['target_col']].median()
    cat_mean_table = df.groupby('Component Category')[cfg['target_col']].mean()
    cat_median_table = df.groupby('Component Category')[cfg['target_col']].median()
    cat_count_table = df.groupby('Component Category')[cfg['target_col']].count()

    baseline_rows = [{
        'Component Category': '*** GLOBAL ***',
        'Baseline_Mean_Mass': global_mean,
        'Baseline_Median_Mass': global_median,
        'Sample_Count': len(df),
    }]
    for cat in sorted(cat_mean_table.index):
        baseline_rows.append({
            'Component Category': cat,
            'Baseline_Mean_Mass': cat_mean_table[cat],
            'Baseline_Median_Mass': cat_median_table[cat],
            'Sample_Count': cat_count_table[cat],
        })
    df_baseline_values = pd.DataFrame(baseline_rows)
    df_baseline_values.to_csv(cfg['save_baseline_values_csv'], index=False)
    print(f"Saved baseline values to {cfg['save_baseline_values_csv']}")

    # ------------------------------------------------------------------
    # 4. Compute SBERT embeddings
    # ------------------------------------------------------------------
    print("\nComputing SBERT embeddings...")
    sbert = SentenceTransformer(cfg['embedding_model'])
    texts = df[cfg['text_col']].astype(str).tolist()
    X_raw = sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    X_pred_raw = None
    if df_pred is not None:
        texts_pred = df_pred[cfg['text_col']].astype(str).tolist()
        X_pred_raw = sbert.encode(texts_pred, show_progress_bar=True, convert_to_numpy=True)

    y = df[cfg['target_col']].values.astype(float)

    # ------------------------------------------------------------------
    # 5. Optuna tuning (optional)
    # ------------------------------------------------------------------
    best_nn_params = None
    if cfg.get('use_optuna', False):
        print("\n" + "=" * 80)
        print("OPTUNA HYPERPARAMETER TUNING")
        print("=" * 80)
        try:
            sampler = TPESampler(seed=cfg.get('random_seed', 42))
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
            )
            study.optimize(
                lambda trial: optuna_objective(trial, cfg, X_raw, y, df),
                n_trials=cfg.get('optuna_n_trials', 40),
                timeout=cfg.get('optuna_timeout_s'),
                show_progress_bar=True,
                n_jobs=1,
            )
            best_nn_params = study.best_params
            best_nn_params['epochs'] = cfg['epochs']
            best_nn_params['early_stopping_patience'] = cfg['early_stopping_patience']

            print(f"\nBest Trial R²: {study.best_value:.4f}")
            print("Best Hyperparameters:")
            for param, value in best_nn_params.items():
                if param not in ('epochs', 'early_stopping_patience'):
                    print(f"  {param}: {value}")
            study.trials_dataframe().to_csv('optuna_trials.csv', index=False)
        except Exception as e:
            print(f"Optuna tuning failed: {e}")
            best_nn_params = None

    # ------------------------------------------------------------------
    # 6. K-Fold cross-validation
    # ------------------------------------------------------------------
    n_folds = cfg.get('cv_folds', 10)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.get('random_seed', 42))

    metrics_nn = []
    metrics_baseline_global_mean = []
    metrics_baseline_global_median = []
    metrics_baseline_cat_mean = []
    metrics_baseline_cat_median = []
    preds_nn_on_predict = []
    last_fold_data = {}
    last_fold_history = None

    print(f"\nRunning {n_folds}-fold cross-validation...")

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_raw), 1):
        print(f"FOLD {fold}/{n_folds}")
        set_seed(cfg.get('random_seed', 42) + fold)

        X_train_raw, X_test_raw = X_raw[tr_idx], X_raw[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

        # Scale features within fold to prevent data leakage
        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        X_pred = scaler.transform(X_pred_raw) if X_pred_raw is not None else None

        # --- Baseline: Global Mean ---
        baseline_pred = np.full(len(y_test), np.mean(y_train))
        metrics_baseline_global_mean.append({
            'fold': fold,
            'R2': r2(y_test, baseline_pred),
            'MAE': mae(y_test, baseline_pred),
            'RMSE': rmse(y_test, baseline_pred),
        })

        # --- Baseline: Global Median ---
        baseline_median_pred = np.full(len(y_test), np.median(y_train))
        metrics_baseline_global_median.append({
            'fold': fold,
            'R2': r2(y_test, baseline_median_pred),
            'MAE': mae(y_test, baseline_median_pred),
            'RMSE': rmse(y_test, baseline_median_pred),
        })

        # --- Baseline: Category Mean/Median ---
        train_df = df.iloc[tr_idx]
        test_df = df.iloc[te_idx]
        cat_train = train_df['Component Category']
        cat_test = test_df['Component Category']
        y_train_series = train_df[cfg['target_col']]

        cat_mean_fold = y_train_series.groupby(cat_train).mean()
        cat_median_fold = y_train_series.groupby(cat_train).median()

        baseline_cat_mean_pred = cat_test.map(cat_mean_fold).fillna(y_train_series.mean())
        baseline_cat_median_pred = cat_test.map(cat_median_fold).fillna(y_train_series.median())

        metrics_baseline_cat_mean.append({
            'fold': fold,
            'R2': r2(y_test, baseline_cat_mean_pred),
            'MAE': mae(y_test, baseline_cat_mean_pred),
            'RMSE': rmse(y_test, baseline_cat_mean_pred),
        })
        metrics_baseline_cat_median.append({
            'fold': fold,
            'R2': r2(y_test, baseline_cat_median_pred),
            'MAE': mae(y_test, baseline_cat_median_pred),
            'RMSE': rmse(y_test, baseline_cat_median_pred),
        })

        # --- Train Neural Network ---
        model_nn, inv_transform, history = train_nn(
            cfg, X_train, y_train, X_test, y_test,
            params_override=best_nn_params,
        )

        # Predict on test set
        test_loader = DataLoader(EmbeddingDataset(X_test),
                                 batch_size=cfg.get('batch_size', 128))
        preds_nn = predict(model_nn, test_loader, cfg['device'])
        if cfg.get('target_normalization') == 'log':
            preds_nn = inv_transform(preds_nn)

        metrics_nn.append({
            'fold': fold,
            'R2': r2(y_test, preds_nn),
            'MAE': mae(y_test, preds_nn),
            'RMSE': rmse(y_test, preds_nn),
        })

        # Predict on prediction set
        if X_pred is not None:
            pred_loader = DataLoader(EmbeddingDataset(X_pred),
                                     batch_size=cfg.get('batch_size', 128))
            preds_pred = predict(model_nn, pred_loader, cfg['device'])
            if cfg.get('target_normalization') == 'log':
                preds_pred = inv_transform(preds_pred)
            preds_nn_on_predict.append(preds_pred)
        else:
            preds_nn_on_predict.append(None)

        # Store last fold data for plotting
        if fold == n_folds:
            last_fold_data['NN'] = {'y_true': y_test, 'y_pred': preds_nn}
            last_fold_history = history

    # ------------------------------------------------------------------
    # 7. Aggregate metrics
    # ------------------------------------------------------------------
    def build_metrics_row(model_name, metrics_list):
        """Build a summary row with mean +/- std for each metric."""
        r2_s = summarise_metric([m['R2'] for m in metrics_list])
        mae_s = summarise_metric([m['MAE'] for m in metrics_list])
        rmse_s = summarise_metric([m['RMSE'] for m in metrics_list])
        return {
            'model': model_name,
            'R2_mean': r2_s['mean'], 'R2_std': r2_s['std'],
            'MAE_mean': mae_s['mean'], 'MAE_std': mae_s['std'],
            'RMSE_mean': rmse_s['mean'], 'RMSE_std': rmse_s['std'],
        }

    rows = [
        build_metrics_row('NN', metrics_nn),
        build_metrics_row('Baseline_GlobalMean', metrics_baseline_global_mean),
        build_metrics_row('Baseline_GlobalMedian', metrics_baseline_global_median),
        build_metrics_row('Baseline_CategoryMean', metrics_baseline_cat_mean),
        build_metrics_row('Baseline_CategoryMedian', metrics_baseline_cat_median),
    ]

    df_metrics = (pd.DataFrame(rows)
                  .sort_values('R2_mean', ascending=False)
                  .reset_index(drop=True))
    df_metrics.to_csv(cfg['save_r2_summary_csv'], index=False)
    print(f"\nSaved metrics summary to {cfg['save_r2_summary_csv']}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY (mean ± std across folds)")
    print(f"{'=' * 80}")
    for _, row in df_metrics.iterrows():
        print(f"\n{row['model']}:")
        print(f"  R²:   {row['R2_mean']:.4f} ± {row['R2_std']:.4f}")
        print(f"  MAE:  {row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}")
        print(f"  RMSE: {row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}")

    # ------------------------------------------------------------------
    # 8. Save predictions on external set
    # ------------------------------------------------------------------
    if df_pred is not None:
        pred_df = df_pred.copy().reset_index(drop=True)
        nn_mean, nn_std = aggregate_predictions(preds_nn_on_predict)
        if nn_mean is not None:
            pred_df['NN_mean'] = nn_mean
            pred_df['NN_std'] = nn_std
        pred_df.to_csv(cfg['save_pred_stats_csv'], index=False)
        print(f"Saved prediction stats to {cfg['save_pred_stats_csv']}")

    # ------------------------------------------------------------------
    # 9. Save plots
    # ------------------------------------------------------------------
    save_dir = cfg.get('save_plots_dir', 'plots')
    os.makedirs(save_dir, exist_ok=True)

    if 'NN' in last_fold_data:
        yt = last_fold_data['NN']['y_true']
        yp = last_fold_data['NN']['y_pred']
        save_hexbin_plot(yt, yp, save_dir, 'NN', target_col=cfg['target_col'])
        save_residual_histogram(yt, yp, save_dir, 'NN')

    if last_fold_history is not None:
        save_learning_curve(last_fold_history, save_dir, 'NN')

    print(f"\nAll outputs saved to {save_dir}/")
    print("Done.")

    return {
        'metrics_summary': df_metrics,
        'baseline_values': df_baseline_values,
        'dataset_summary': df_dataset_summary,
        'dataset_by_category': df_dataset_by_category,
    }


if __name__ == '__main__':
    out = main(CONFIG)
