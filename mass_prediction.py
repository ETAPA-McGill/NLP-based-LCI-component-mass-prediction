"""
Mass prediction from component description text.

Text encoders (SBERT / TF-IDF) + regression heads, compared with K-fold
cross-validation against naive baselines. Outputs go to output_dir/figures
and output_dir/tables. See the GitHub repository for full documentation.
"""

import os
import math
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

matplotlib.use('Agg')
sns.set_style('whitegrid')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # --- Input data ---
    'train_csv': 'CSM - Master - EC - DD.csv',
    'predict_csv': None,   # None to skip
    'text_col': 'Name',
    'target_col': 'Mass (g)',
    'category_col': 'Component Category',
    'drop_unnamed_columns': True,

    # Merge rows with identical text (target averaged); runs before filtering.
    'deduplicate': True,

    # --- Outlier filtering: None | 'iqr' | 'percentile' | 'mass_cutoff' ---
    'outlier_method': 'mass_cutoff',
    'iqr_multiplier': 3.0,
    'percentile_lower': 0.0,
    'percentile_upper': 99,
    'mass_cutoff_g': 25.0,            # drop anything heavier than this (grams)

    # --- Encoders: 'sbert', 'tfidf', 'tfidf_char', or combos like 'sbert+tfidf_char' ---
    'encoders': ['sbert+tfidf_char'],
    'embedders': {
        'sbert': {'hf': 'all-mpnet-base-v2', 'prefix': '', 'normalize': False},
    },
    'encoder_params': {
        'svd_components': 300,          # LSA dimensions for the sparse encoders
        'tfidf_max_features': 40000,
        'tfidf_ngram_range': (1, 2),
        'char_ngram_range': (2, 5),
    },

    # --- Models: 'mlp', 'linear', 'ridge', 'random_forest', 'knn', 'hist_gbr' ---
    'models': ['mlp'],
    'model_params': {
        'ridge_alpha': 1.0,
        'rf_n_estimators': 200,
        'rf_max_depth': None,
        'knn_n_neighbors': 10,
        'knn_weights': 'distance',
        'hgb_max_iter': 300,
        'hgb_learning_rate': 0.1,
    },

    # --- MLP training ---
    'batch_size': 128,
    'epochs': 200,
    'lr': 0.00015,
    'weight_decay': 0.03,
    'grad_clip_norm': 1.0,             # gradient-norm clip (None to disable)
    'lr_scheduler': 'cosine',          # 'cosine' | None
    'hidden_dim': 512,
    'mlp_hidden_layers': 2,
    'dropout': 0.1,
    'early_stopping_patience': 40,
    'early_stopping_val_fraction': 0.15,  # carved from the train fold only
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'loss_function': 'huber',          # 'mse' | 'mae' | 'huber'
    'huber_delta': 1.5,
    # None | 'log1p' | 'log' | 'cbrt' | 'boxcox' (metrics always in grams)
    'target_normalization': None,

    # --- Error analysis ---
    'mass_bin_edges': [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
    'hexbin_zoom_range': (0.0, 1.0),   # second hexbin zoomed to this window (g)

    # --- Embedding-space analysis (t-SNE + silhouette) ---
    'tsne_plot': True,
    'tsne_collapse_other': True,       # fold minor categories into 'Other'
    'tsne_perplexity': 30,
    'tsne_n_iter': 1000,
    'silhouette_analysis': True,
    'max_silhouette_samples': 8000,

    # --- Cross-validation ---
    'cv_folds': 10,
    'random_seed': 42,
    'experiment_sample_size': None,    # subsample for quick runs; None = all

    # --- Optuna (tunes the MLP only) ---
    'use_optuna': False,
    'optuna_n_trials': 50,
    'optuna_timeout_s': None,
    'optuna_cv_folds': 3,              # inner CV folds per trial
    'optuna_target_norms': [None, 'log', 'log1p', 'cbrt', 'boxcox'],

    # --- Outputs ---
    'output_dir': '.',
    'figures_dir': 'figures',
    'tables_dir': 'tables',

    'verbose': True,
}

# Fixed output filenames (created inside the tables/ folder).
TABLE_FILES = {
    'model_comparison':       'model_comparison.csv',
    'dataset_summary':        'dataset_summary.csv',
    'dataset_by_category':    'dataset_by_category.csv',
    'error_by_category':      'error_by_category.csv',
    'error_by_mass_range':    'error_by_mass_range.csv',
    'silhouette_by_category': 'silhouette_by_category.csv',
    'predictions':            'predictions.csv',
}

# =============================================================================
# LOGGING AND GENERAL UTILITIES
# =============================================================================
def log(msg=''):
    """Timestamped status line."""
    if msg:
        print(f"[{datetime.now():%H:%M:%S}] {msg}")
    else:
        print()

def banner(title):
    """Visual section divider for the console output."""
    line = '=' * 78
    print(f"\n{line}\n {title}\n{line}")

def set_seed(seed):
    """Seed every RNG in play so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def figures_dir(cfg):
    """Absolute path of the figures folder (created if missing)."""
    path = os.path.join(os.path.abspath(cfg.get('output_dir', '.')),
                        cfg.get('figures_dir', 'figures'))
    os.makedirs(path, exist_ok=True)
    return path

def tables_dir(cfg):
    """Absolute path of the tables folder (created if missing)."""
    path = os.path.join(os.path.abspath(cfg.get('output_dir', '.')),
                        cfg.get('tables_dir', 'tables'))
    os.makedirs(path, exist_ok=True)
    return path

def save_table(df, cfg, name):
    """Write a DataFrame to tables/<name> and log it."""
    path = os.path.join(tables_dir(cfg), TABLE_FILES[name])
    df.to_csv(path, index=False)
    log(f"Saved {TABLE_FILES[name]}  ({len(df)} rows)")
    return path

# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================
def has_predict_csv(cfg):
    """True if a usable prediction CSV path is configured."""
    val = cfg.get('predict_csv')
    if val is None:
        return False
    if isinstance(val, str) and val.strip().upper() == 'NONE':
        return False
    return True

def outlier_mask(series, cfg):
    """(boolean keep-mask, description) for the configured outlier method."""
    method = cfg.get('outlier_method')
    if method == 'iqr':
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        mult = cfg.get('iqr_multiplier', 1.5)
        lower, upper = q1 - mult * (q3 - q1), q3 + mult * (q3 - q1)
        return series.between(lower, upper), f"IQR, multiplier={mult}"
    if method == 'percentile':
        lo = float(cfg.get('percentile_lower', 1.0))
        hi = float(cfg.get('percentile_upper', 99.0))
        if not (0 <= lo < hi <= 100):
            return pd.Series(True, index=series.index), None
        lower, upper = np.percentile(series, lo), np.percentile(series, hi)
        return series.between(lower, upper), f"percentile {lo}-{hi}"
    if method == 'mass_cutoff':
        cutoff = float(cfg.get('mass_cutoff_g', 25.0))
        return series <= cutoff, f"mass <= {cutoff} g"
    return pd.Series(True, index=series.index), None

def normalize_text(series):
    """Canonical form for duplicate/overlap matching: strip + lowercase."""
    return series.astype(str).str.strip().str.lower()

def deduplicate(df, cfg):
    """Merge rows sharing the same normalized text (target averaged)."""
    text_col = cfg['text_col']
    target = cfg['target_col']
    key = normalize_text(df[text_col])
    n_dupes = int(key.duplicated().sum())
    if n_dupes == 0:
        return df

    agg = {text_col: 'first'}
    for col in df.columns:
        if col == text_col:
            continue
        if col == target:
            agg[col] = 'mean'
        else:
            agg[col] = 'first'

    original_columns = list(df.columns)
    df = df.copy()
    df['_key'] = key
    merged = df.groupby('_key', sort=False).agg(agg).reset_index(drop=True)
    merged = merged[original_columns]
    log(f"Deduplicated {n_dupes} rows ({len(df)} -> {len(merged)})")
    return merged

def load_data(cfg):
    """Load, clean, deduplicate, and filter the training data; optionally load
    the prediction set (overlapping training texts are removed)."""
    df = pd.read_csv(cfg['train_csv'])

    if cfg.get('drop_unnamed_columns', True):
        unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
        if unnamed:
            df = df.drop(columns=unnamed)

    df = df.dropna(subset=[cfg['text_col'], cfg['target_col']]).reset_index(drop=True)

    if cfg.get('deduplicate', True):
        df = deduplicate(df, cfg)

    df_before = df.copy()
    keep, desc = outlier_mask(df[cfg['target_col']], cfg)
    if desc is not None:
        n0 = len(df)
        df = df[keep].reset_index(drop=True)
        log(f"Outlier filter [{desc}]: removed {n0 - len(df)} rows -> {len(df)} remain")
    df_after = df.copy()

    df_pred = None
    if has_predict_csv(cfg):
        try:
            df_pred = pd.read_csv(cfg['predict_csv'])
            df_pred = df_pred.dropna(subset=[cfg['text_col']]).reset_index(drop=True)
            pred_keys = set(normalize_text(df_pred[cfg['text_col']]))
            overlap = normalize_text(df[cfg['text_col']]).isin(pred_keys)
            if overlap.any():
                df = df[~overlap].reset_index(drop=True)
            log(f"Loaded {len(df_pred)} components to predict "
                f"({int(overlap.sum())} overlapping training rows removed)")
        except Exception as e:
            log(f"WARNING: could not load prediction CSV ({e}); continuing without it")
            df_pred = None

    return df, df_pred, df_before, df_after

# =============================================================================
# DATASET SUMMARIES (formatted like the manuscript tables)
# =============================================================================
def mass_stats_row(label, series):
    """One row of mass statistics in the manuscript's column layout."""
    return {'Dataset': label,
            'N': len(series),
            'Mean (g)': series.mean(),
            'Median (g)': series.median(),
            'Std Dev (g)': series.std(),
            'Max (g)': series.max()}

def dataset_summary(df_before, df_after, df_final, cfg):
    """Whole-dataset mass statistics at each cleaning stage, one row per stage."""
    target = cfg['target_col']
    rows = [mass_stats_row('After deduplication', df_before[target]),
            mass_stats_row('After outlier filter', df_after[target]),
            mass_stats_row('Final training set', df_final[target])]
    return pd.DataFrame(rows)

def category_stats(df, cfg):
    """Per-category mass statistics in the layout of manuscript Table 4:
    a whole-dataset row first, then every category in alphabetical order."""
    target = cfg['target_col']
    cat_col = cfg.get('category_col', 'Component Category')
    rows = [mass_stats_row('Whole Dataset', df[target])]
    for cat in sorted(df[cat_col].astype(str).unique()):
        rows.append(mass_stats_row(cat, df.loc[df[cat_col] == cat, target]))
    return pd.DataFrame(rows)

# =============================================================================
# TEXT ENCODERS
# =============================================================================
ENCODER_LABELS = {
    'sbert': 'SBERT', 'tfidf': 'TF-IDF', 'tfidf_char': 'CharTFIDF',
}

def encoder_label(name):
    """Readable label for an encoder string, including '+' combos."""
    parts = []
    for p in name.split('+'):
        parts.append(ENCODER_LABELS.get(p, p))
    return '+'.join(parts)

def parse_encoder(name, cfg):
    """Split an encoder string into (embedder | None, sparse | None)."""
    reg = cfg.get('embedders', {})
    emb = sparse = None
    for part in name.split('+'):
        if part in reg:
            emb = part
        else:
            sparse = part
    return emb, sparse

def embedders_used(encoders, cfg):
    """Set of transformer embedder keys referenced by the encoder list."""
    reg = cfg.get('embedders', {})
    used = set()
    for encoder in encoders:
        for part in encoder.split('+'):
            if part in reg:
                used.add(part)
    return used

def make_vectorizer(name, cfg):
    """(vectorizer, TruncatedSVD) pair for a sparse encoder; fit on train only."""
    p = cfg.get('encoder_params', {})
    seed = cfg.get('random_seed', 42)
    if name == 'tfidf':
        vec = TfidfVectorizer(max_features=p.get('tfidf_max_features', 20000),
                              ngram_range=tuple(p.get('tfidf_ngram_range', (1, 2))),
                              lowercase=True)
    elif name == 'tfidf_char':
        vec = TfidfVectorizer(analyzer='char_wb',
                              ngram_range=tuple(p.get('char_ngram_range', (2, 5))),
                              max_features=p.get('tfidf_max_features', 20000),
                              lowercase=True)
    else:
        raise ValueError(f"Unknown text encoder: {name!r}")
    svd = TruncatedSVD(n_components=p.get('svd_components', 300), random_state=seed)
    return vec, svd

def sparse_features(name, cfg, train_texts, test_texts, predict_texts):
    """Fit sparse encoder + SVD on the train fold; return dense matrices."""
    vec, svd = make_vectorizer(name, cfg)
    Xtr_sparse = vec.fit_transform(train_texts)
    safe = max(1, min(svd.n_components, min(Xtr_sparse.shape) - 1))
    if safe != svd.n_components:
        svd.n_components = safe
    Xtr = svd.fit_transform(Xtr_sparse)
    Xte = svd.transform(vec.transform(test_texts))
    Xpred = (svd.transform(vec.transform(predict_texts))
             if predict_texts is not None else None)
    return Xtr, Xte, Xpred

def encode_fold(name, cfg, train_texts, test_texts, precomputed=None,
                tr_idx=None, te_idx=None, predict_texts=None):
    """(X_train, X_test, X_predict) for one encoder string. Embedder blocks are
    precomputed (fixed pretrained transforms); sparse blocks fit on train only."""
    emb, sparse = parse_encoder(name, cfg)
    tr_blocks, te_blocks, pred_blocks = [], [], []
    have_pred = predict_texts is not None

    if emb is not None:
        tr_blocks.append(precomputed[emb][tr_idx])
        te_blocks.append(precomputed[emb][te_idx])
        ep = precomputed.get(emb + '_predict')
        if ep is None:
            have_pred = False
        else:
            pred_blocks.append(ep)

    if sparse is not None:
        s_tr, s_te, s_pred = sparse_features(sparse, cfg, train_texts, test_texts,
                                           predict_texts)
        tr_blocks.append(s_tr)
        te_blocks.append(s_te)
        if s_pred is None:
            have_pred = False
        else:
            pred_blocks.append(s_pred)

    if not tr_blocks:
        raise ValueError(f"Unknown encoder: {name!r}")

    def stack(blocks):
        if len(blocks) == 1:
            return blocks[0]
        return np.hstack(blocks)

    X_train = stack(tr_blocks)
    X_test = stack(te_blocks)
    if have_pred and pred_blocks:
        X_pred = stack(pred_blocks)
    else:
        X_pred = None
    return X_train, X_test, X_pred

# =============================================================================
# METRICS
# =============================================================================
def r2(y, p):
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return r2_score(y, p)
    except Exception:
        return np.nan

def mae(y, p):
    try:
        return mean_absolute_error(y, p)
    except Exception:
        return np.nan

def rmse(y, p):
    try:
        return math.sqrt(mean_squared_error(y, p))
    except Exception:
        return np.nan

def mape(y, p):
    """Mean absolute percentage error (%), guarded against near-zero masses."""
    try:
        y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
        denom = np.clip(np.abs(y), 1e-9, None)
        return float(np.mean(np.abs(y - p) / denom) * 100.0)
    except Exception:
        return np.nan

def aggregate_metric(values):
    """Mean and sample standard deviation across folds, NaN-safe."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return {'mean': np.nanmean(arr) if len(arr) else np.nan, 'std': np.nan}
    return {'mean': np.mean(arr), 'std': np.std(arr, ddof=1)}

def fold_metrics(fold, y_true, y_pred):
    """Per-fold metric record for one model."""
    return {'fold': fold, 'R2': r2(y_true, y_pred),
            'MAE': mae(y_true, y_pred), 'RMSE': rmse(y_true, y_pred),
            'MAPE': mape(y_true, y_pred)}

def per_group_metrics(fold, y_true, y_pred, groups):
    """Per-group MAE/RMSE records for one fold (category or mass-range)."""
    groups, y_pred = np.asarray(groups), np.asarray(y_pred)
    records = []
    for g in np.unique(groups):
        m = groups == g
        records.append({'fold': fold, 'group': g, 'n': int(m.sum()),
                        'MAE': mae(y_true[m], y_pred[m]),
                        'RMSE': rmse(y_true[m], y_pred[m])})
    return records

def build_combo_row(combo, encoder, model, metrics_list):
    """Aggregate one combo's per-fold metrics into a mean +/- std row."""
    row = {'combo': combo, 'encoder': encoder, 'model': model}
    for key in ('R2', 'MAE', 'RMSE', 'MAPE'):
        s = aggregate_metric([m[key] for m in metrics_list])
        row[f'{key}_mean'], row[f'{key}_std'] = s['mean'], s['std']
    return row

def aggregate_pergroup(pergroup):
    """Aggregate per-group fold records into one summary DataFrame per combo:
    {(encoder, model): DataFrame indexed by group with mean/std columns}."""
    out = {}
    for (e, m), recs in pergroup.items():
        if not recs:
            continue
        dfp = pd.DataFrame(recs)
        rows = []
        for grp, g in dfp.groupby('group'):
            row = {'group': grp, 'n_mean': float(g['n'].mean())}
            for key in ('MAE', 'RMSE'):
                row[f'{key}_mean'] = float(np.nanmean(g[key]))
                row[f'{key}_std'] = float(np.nanstd(g[key]))
            rows.append(row)
        out[(e, m)] = pd.DataFrame(rows).sort_values('group').reset_index(drop=True)
    return out

def pergroup_table(agg, combos, group_header, group_order=None):
    """Manuscript-style error table (Tables 6/7); a Model column is added only
    when more than one encoder+model combination was run."""
    frames = []
    for (e, m) in combos:
        if (e, m) not in agg:
            continue
        d = agg[(e, m)].copy()
        if group_order is not None:
            order = {g: i for i, g in enumerate(group_order)}
            d = d.sort_values(by='group', key=lambda s: s.map(order))
        table = pd.DataFrame({
            group_header: d['group'],
            'Mean Count (n)': d['n_mean'],
            'MAE': d['MAE_mean'], 'MAE_std': d['MAE_std'],
            'RMSE': d['RMSE_mean'], 'RMSE_std': d['RMSE_std'],
        })
        if len(combos) > 1:
            table.insert(0, 'Model', combo_label(e, m))
        frames.append(table)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())

def mass_bin_labels(y, edges):
    """Bin each mass into a labelled range ('0.0 - 0.1'); returns
    (labels, ordered names). A final open-ended bin is appended."""
    edges = sorted(float(e) for e in edges)
    bounds = []
    for i in range(len(edges) - 1):
        bounds.append((edges[i], edges[i + 1]))
    bounds.append((edges[-1], np.inf))

    names = []
    for lo, hi in bounds:
        if np.isinf(hi):
            names.append(f"> {lo:.1f}")
        else:
            names.append(f"{lo:.1f} - {hi:.1f}")

    idx = np.clip(np.digitize(y, edges) - 1, 0, len(bounds) - 1)
    labels = [names[i] for i in idx]
    return np.array(labels), names

# =============================================================================
# BASELINES
# =============================================================================
BASELINE_NAMES = ['Baseline_GlobalMean', 'Baseline_GlobalMedian',
                  'Baseline_CategoryMean', 'Baseline_CategoryMedian']

def baseline_predictions(y_train, train_df, test_df, cfg):
    """The four naive baseline prediction arrays for one fold."""
    cat_col, target = cfg.get('category_col', 'Component Category'), cfg['target_col']
    cat_train, cat_test = train_df[cat_col], test_df[cat_col]
    y_series = train_df[target]
    return {
        'Baseline_GlobalMean':     np.full(len(test_df), np.mean(y_train)),
        'Baseline_GlobalMedian':   np.full(len(test_df), np.median(y_train)),
        'Baseline_CategoryMean':   cat_test.map(y_series.groupby(cat_train).mean())
                                           .fillna(y_series.mean()).values,
        'Baseline_CategoryMedian': cat_test.map(y_series.groupby(cat_train).median())
                                           .fillna(y_series.median()).values,
    }

# =============================================================================
# REGRESSION MODELS (scikit-learn)
# =============================================================================
MODEL_LABELS = {
    'mlp': 'MLP', 'linear': 'Linear', 'ridge': 'Ridge',
    'random_forest': 'RandomForest', 'knn': 'KNN', 'hist_gbr': 'HistGBR',
}

def combo_label(encoder, model):
    """Human-readable name for an encoder+model combination."""
    return f"{encoder_label(encoder)}+{MODEL_LABELS.get(model, model)}"

def make_sklearn_model(name, cfg):
    """Construct a scikit-learn regressor with the configured parameters."""
    p = cfg.get('model_params', {})
    seed = cfg.get('random_seed', 42)
    if name == 'linear':
        return LinearRegression()
    if name == 'ridge':
        return Ridge(alpha=p.get('ridge_alpha', 1.0))
    if name == 'random_forest':
        return RandomForestRegressor(n_estimators=p.get('rf_n_estimators', 200),
                                     max_depth=p.get('rf_max_depth', None),
                                     n_jobs=-1, random_state=seed)
    if name == 'knn':
        return KNeighborsRegressor(n_neighbors=p.get('knn_n_neighbors', 10),
                                   weights=p.get('knn_weights', 'distance'),
                                   n_jobs=-1)
    if name == 'hist_gbr':
        return HistGradientBoostingRegressor(max_iter=p.get('hgb_max_iter', 300),
                                             learning_rate=p.get('hgb_learning_rate', 0.1),
                                             random_state=seed)
    raise ValueError(f"Unknown sklearn model: {name!r}")

# =============================================================================
# TARGET TRANSFORMS
# =============================================================================
_EPS = 1e-9

def make_target_transform(name, y_train):
    """(forward, inverse, positive_space) target transform, fit on train only.
    positive_space=True permits the Softplus output head."""
    if not name:
        name = 'none'
    elif isinstance(name, str):
        name = name.lower()

    if name in (None, 'none'):
        def forward(y):
            return np.asarray(y, dtype=float)

        def inverse(z):
            return np.asarray(z, dtype=float)

        return forward, inverse, True

    if name == 'log1p':
        return np.log1p, np.expm1, True

    if name == 'log':
        def forward(y):
            return np.log(np.clip(y, _EPS, None))

        return forward, np.exp, False

    if name == 'cbrt':
        def inverse(z):
            return np.power(np.clip(z, 0.0, None), 3)

        return np.cbrt, inverse, True

    if name == 'boxcox':
        pt = PowerTransformer(method='box-cox', standardize=True)
        y_col = np.clip(np.asarray(y_train, dtype=float), _EPS, None).reshape(-1, 1)
        pt.fit(y_col)

        def forward(y):
            y_col = np.clip(np.asarray(y, dtype=float), _EPS, None).reshape(-1, 1)
            return pt.transform(y_col).ravel()

        def inverse(z):
            z_col = np.asarray(z, dtype=float).reshape(-1, 1)
            return np.clip(pt.inverse_transform(z_col).ravel(), 0.0, None)

        return forward, inverse, False

    raise ValueError(f"Unknown target_normalization: {name!r}")

def train_predict_sklearn(name, cfg, X_train, y_train, X_test, X_predict=None):
    """Fit a scikit-learn model; return (test_preds, predict_preds) in grams."""
    model = make_sklearn_model(name, cfg)
    fwd, inv, _ = make_target_transform(cfg.get('target_normalization'), y_train)
    model.fit(X_train, fwd(y_train))
    test_pred = inv(model.predict(X_test))
    pred_pred = inv(model.predict(X_predict)) if X_predict is not None else None
    return test_pred, pred_pred

# =============================================================================
# NEURAL NETWORK
# =============================================================================
class EmbeddingDataset(Dataset):
    """Feature vectors with optional targets."""

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class MLPRegressor(nn.Module):
    """MLP; layer i has width hidden_dim // 2**i (LeakyReLU, Dropout, BatchNorm),
    with an optional Softplus head to keep predictions positive."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.2,
                 n_hidden_layers=2, positive_output=True):
        super().__init__()
        layers, in_dim = [], input_dim
        for i in range(max(1, n_hidden_layers)):
            out_dim = max(8, hidden_dim // (2 ** i))
            layers += [nn.Linear(in_dim, out_dim), nn.LeakyReLU(),
                       nn.Dropout(dropout), nn.BatchNorm1d(out_dim)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        if positive_output:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def build_mlp(input_dim, params):
    """Construct the MLP regressor from a params dict."""
    return MLPRegressor(input_dim, hidden_dim=params['hidden_dim'],
                        dropout=params['dropout'],
                        n_hidden_layers=params.get('mlp_hidden_layers', 2),
                        positive_output=params.get('positive_output', True))

def get_loss_fn(name, huber_delta=1.0):
    """Loss function by name."""
    if name == 'mse':
        return nn.MSELoss()
    if name == 'mae':
        return nn.L1Loss()
    if name == 'huber':
        return nn.SmoothL1Loss(beta=huber_delta)
    return nn.MSELoss()

def make_loader(X, y, batch_size, shuffle):
    """Build a DataLoader over the features (and targets if given)."""
    ds = EmbeddingDataset(X, y)
    # A final batch of one sample crashes BatchNorm, so drop it while training.
    drop_last = False
    if shuffle and len(ds) % batch_size == 1:
        drop_last = True
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)

def train_epoch(model, loader, optimizer, device, loss_fn, grad_clip=None):
    """One training epoch; returns the mean batch loss."""
    model.train()
    losses = []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(Xb), yb)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses) if losses else np.nan

def eval_epoch(model, loader, device):
    """Evaluate; returns (predictions, true values)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.extend(model(Xb.to(device)).cpu().numpy())
            trues.extend(yb.numpy())
    return np.array(preds), np.array(trues)

def predict(model, loader, device):
    """Predictions for an unlabelled dataset."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                Xb = batch[0]
            else:
                Xb = batch
            out.extend(model(Xb.to(device)).cpu().numpy())
    return np.array(out)

def train_mlp(cfg, X_train, y_train, X_val, y_val, params_override=None):
    """Train the MLP with early stopping; returns (best model, inverse
    target transform, per-epoch history)."""
    params = {
        'lr': cfg['lr'],
        'weight_decay': cfg.get('weight_decay', 1e-2),
        'hidden_dim': cfg['hidden_dim'],
        'dropout': cfg['dropout'],
        'mlp_hidden_layers': cfg.get('mlp_hidden_layers', 2),
        'epochs': cfg['epochs'],
        'batch_size': cfg['batch_size'],
        'early_stopping_patience': cfg['early_stopping_patience'],
        'loss_function': cfg['loss_function'],
        'huber_delta': cfg.get('huber_delta', 1.0),
        'target_normalization': cfg.get('target_normalization'),
    }
    if params_override:
        params.update(params_override)

    fwd, inv, positive_space = make_target_transform(
        params['target_normalization'], y_train)
    params['positive_output'] = positive_space

    train_loader = make_loader(X_train, fwd(y_train), params['batch_size'],
                                shuffle=True)
    val_loader = make_loader(X_val, fwd(y_val), params['batch_size'],
                              shuffle=False)

    device = cfg['device']
    model = build_mlp(X_train.shape[1], params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'],
                                  weight_decay=params['weight_decay'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=params['epochs'])
                 if cfg.get('lr_scheduler') == 'cosine' else None)
    loss_fn = get_loss_fn(params['loss_function'], huber_delta=params['huber_delta'])
    grad_clip = cfg.get('grad_clip_norm')

    best_state, best_r2, no_imp = None, -np.inf, 0
    history = {'train_loss': [], 'val_r2': [], 'val_mae': []}

    for _ in tqdm(range(params['epochs']), desc='NN epochs', leave=False):
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 loss_fn, grad_clip)
        if scheduler is not None:
            scheduler.step()
        p_norm, t_norm = eval_epoch(model, val_loader, device)
        p, t = inv(p_norm), inv(t_norm)

        val_r2, val_mae = r2(t, p), mae(t, p)
        history['train_loss'].append(train_loss)
        history['val_r2'].append(val_r2)
        history['val_mae'].append(val_mae)

        if np.isfinite(val_r2) and val_r2 > best_r2:
            best_r2, no_imp = val_r2, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= params['early_stopping_patience']:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, inv, history

# =============================================================================
# TRAIN / PREDICT ONE COMBO
# =============================================================================
def fit_predict_combo(model_name, cfg, X_train, y_train, X_test, X_predict,
                      nn_params, seed):
    """Train one model on one fold; the MLP carves an early-stopping validation
    slice from the training fold. history is None for sklearn models."""
    if model_name == 'mlp':
        frac = cfg.get('early_stopping_val_fraction', 0.15)
        n_val = max(1, int(round(frac * len(X_train))))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(X_train))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]

        model, inv, history = train_mlp(
            cfg, X_train[fit_idx], y_train[fit_idx],
            X_train[val_idx], y_train[val_idx],
            params_override=nn_params)

        bs = cfg.get('batch_size', 128)
        test_preds = inv(predict(model, DataLoader(EmbeddingDataset(X_test),
                                                   batch_size=bs), cfg['device']))
        predict_preds = None
        if X_predict is not None:
            predict_preds = inv(predict(model,
                                        DataLoader(EmbeddingDataset(X_predict),
                                                   batch_size=bs), cfg['device']))
        return test_preds, predict_preds, history

    test_preds, predict_preds = train_predict_sklearn(
        model_name, cfg, X_train, y_train, X_test, X_predict=X_predict)
    return test_preds, predict_preds, None

def aggregate_predictions(pred_list):
    """Per-component mean and std of fold predictions."""
    valid = [p for p in pred_list if p is not None]
    if not valid:
        return None, None
    mat = np.vstack(valid)
    return mat.mean(axis=0), mat.std(axis=0)

# =============================================================================
# HYPERPARAMETER TUNING (OPTUNA)
# =============================================================================
def optuna_objective(trial, cfg, X, y):
    """Mean inner-CV R² (in grams) of an MLP for one sampled configuration."""
    norm_options = cfg.get('optuna_target_norms', [None, 'log1p', 'cbrt', 'boxcox'])
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 3e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512]),
        'mlp_hidden_layers': trial.suggest_categorical('mlp_hidden_layers', [1, 2, 3]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'huber_delta': trial.suggest_float('huber_delta', 0.1, 2.0, log=True),
        'target_normalization': trial.suggest_categorical(
            'target_normalization', norm_options),
        'batch_size': cfg['batch_size'],
        'epochs': cfg['epochs'],
        'early_stopping_patience': cfg['early_stopping_patience'],
    }

    n_inner = cfg.get('optuna_cv_folds', 3)
    kf_tune = KFold(n_splits=n_inner, shuffle=True,
                    random_state=cfg.get('random_seed', 42))
    fold_r2s = []
    for tune_fold, (tr_idx, val_idx) in enumerate(kf_tune.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_val = scaler.transform(X_tr), scaler.transform(X_val)
        try:
            model, inv, _ = train_mlp(cfg, X_tr, y_tr, X_val, y_val,
                                     params_override=params)
            loader = DataLoader(EmbeddingDataset(X_val),
                                batch_size=params['batch_size'])
            preds = inv(predict(model, loader, cfg['device']))
            score = r2(y_val, preds)
            if not np.isnan(score):
                fold_r2s.append(score)
            trial.report(score, tune_fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception:
            continue
    return np.mean(fold_r2s) if fold_r2s else float('-inf')

def tune_mlp(cfg, X, y):
    """Run Optuna on the MLP; return the best params dict, or None on failure."""
    banner("OPTUNA HYPERPARAMETER TUNING (MLP)")
    try:
        sampler = TPESampler(seed=cfg.get('random_seed', 42))
        study = optuna.create_study(
            direction='maximize', sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))

        def objective(trial):
            return optuna_objective(trial, cfg, X, y)

        study.optimize(objective,
                       n_trials=cfg.get('optuna_n_trials', 40),
                       timeout=cfg.get('optuna_timeout_s'),
                       show_progress_bar=True, n_jobs=1)
        best = study.best_params
        best['epochs'] = cfg['epochs']
        best['early_stopping_patience'] = cfg['early_stopping_patience']
        log(f"Best trial R2: {study.best_value:.4f}  params: {best}")
        return best
    except Exception as ex:
        log(f"Optuna tuning failed: {ex}")
        return None

# =============================================================================
# PLOTTING
# =============================================================================
def setup_plot_style():
    """Publication-style matplotlib defaults (high DPI)."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 1500,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 16,
        'axes.labelsize': 19,
        'axes.titlesize': 21,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
    })

def save_figure(fig, save_dir, name, also_pdf=False):
    """Save a figure as SVG (and optionally PDF) into the figures folder."""
    path = os.path.join(save_dir, f"{name}.svg")
    fig.savefig(path, format='svg', bbox_inches='tight', facecolor='white')
    if also_pdf:
        fig.savefig(os.path.join(save_dir, f"{name}.pdf"),
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log(f"Saved {name}.svg")

def save_learning_curve(history, save_dir, prefix):
    """Three-panel learning curve: training loss, validation R2, validation MAE."""
    setup_plot_style()
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')

    axes[1].plot(epochs, history['val_r2'], 'g-', linewidth=2, label='Validation R²')
    axes[1].axhline(y=max(history['val_r2']), color='r', linestyle='--', alpha=0.5,
                    label=f'Best: {max(history["val_r2"]):.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Validation R²')

    axes[2].plot(epochs, history['val_mae'], color='orange', linewidth=2,
                 label='Validation MAE')
    axes[2].axhline(y=min(history['val_mae']), color='r', linestyle='--', alpha=0.5,
                    label=f'Best: {min(history["val_mae"]):.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('Validation MAE')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    save_figure(fig, save_dir, f"learning_curve_{prefix}")

def save_hexbin_plot(y_true, y_pred, save_dir, prefix, target_col='Mass (g)',
                     mass_range=None):
    """Hexbin panels (a) reported vs predicted, (b) residuals vs reported.
    If mass_range is given, the hexbins are rebuilt over just that window."""
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec

    setup_plot_style()
    CMAP, REF_COLOR, REF_ALPHA = 'viridis_r', '#D3D3D3', 0.6

    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mass_range is not None:
        low, high = mass_range
        valid = valid & (y_true >= low) & (y_true <= high)
    if valid.sum() == 0:
        return
    yt, yp = y_true[valid], y_pred[valid]
    resid = yt - yp

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.30)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Probe panel (a)'s color range so both panels share one scale.
    h_temp = ax1.hexbin(yt, yp, gridsize=50, cmap=CMAP, mincnt=1,
                        edgecolors='none', norm=LogNorm())
    vmin, vmax = h_temp.get_clim()
    ax1.cla()
    shared_norm = LogNorm(vmin=vmin, vmax=vmax)

    ax1.hexbin(yt, yp, gridsize=50, cmap=CMAP, mincnt=1,
               edgecolors='none', norm=shared_norm)
    # Keep axis limits so the reference line can't stretch them.
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax1.plot([mn, mx], [mn, mx], color=REF_COLOR, linewidth=2,
             linestyle='--', alpha=REF_ALPHA)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel(f'Reported {target_col}')
    ax1.set_ylabel(f'Predicted {target_col}')
    ax1.set_title('(a)')
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)

    h2 = ax2.hexbin(yt, resid, gridsize=50, cmap=CMAP, mincnt=1,
                    edgecolors='none', norm=shared_norm)
    ax2.axhline(0, color=REF_COLOR, linestyle='--', linewidth=2, alpha=REF_ALPHA)
    ax2.set_xlabel(f'Reported {target_col}')
    ax2.set_ylabel('Residual (Reported − Predicted)')
    ax2.set_title('(b)')
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.4)

    cb = fig.colorbar(h2, cax=cbar_ax)
    cb.set_label('Count (log scale)')

    if mass_range is None:
        name = f"parity_residuals_{prefix}"
    else:
        low, high = mass_range
        name = f"parity_residuals_{prefix}_zoom_{low:g}-{high:g}g"
    save_figure(fig, save_dir, name)

def save_residual_histogram(y_true, y_pred, save_dir, prefix):
    """Residual histogram with a log y-scale."""
    setup_plot_style()
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if valid.sum() == 0:
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
    save_figure(fig, save_dir, f"residual_histogram_{prefix}")

# =============================================================================
# EMBEDDING-SPACE ANALYSIS (t-SNE + silhouette)
# =============================================================================
# Qualitative palette; 'Other' stays neutral grey and is drawn behind.
CATEGORY_COLORS = {
    'Resistor':            '#E6194B',
    'Capacitor':           '#4363D8',
    'Integrated Circuit':  '#3CB44B',
    'Connector':           '#F58231',
    'Transistor':          '#911EB4',
    'Diode':               '#F032E6',
    'Inductor':            '#42D4F4',
    'Cable':               '#9A6324',
    'Electromechanical':   '#808000',
    'Other':               '#9A9A9A',
}

# Distinct fallbacks for any category not named above.
FALLBACK_COLORS = ['#000075', '#800000', '#469990', '#E69F00',
                   '#F564A9', '#5A5A5A', '#00C2C7', '#7A4900']

# Minor categories collapsed into 'Other' when cfg['tsne_collapse_other'] is on.
OTHER_CATEGORIES = {
    'Acoustic Device', 'Battery and Accessories', 'Battery and Accesories',
    'Crystal / Oscillator', 'Display', 'Energy Generation', 'Filter',
    'Fuse / Protection', 'Magnetics', 'Motion Device', 'Optical', 'Relay',
    'Thermal Management', 'Electrochemical', 'Processor', 'processor',
    'Accessories', 'Sensor', 'Frequency Generation', 'Motor',
    'Protection Device', 'Sensor and Accesories', 'Sensor and Accessories',
    'Switch', 'Wireless',
}

def ordered_categories(categories):
    """Sorted unique categories with 'Other' forced to the front (drawn first)."""
    unique_cats = sorted(set(categories))
    if 'Other' in unique_cats:
        unique_cats.remove('Other')
        unique_cats.insert(0, 'Other')
    return unique_cats

def build_color_map(unique_cats):
    """Assign a distinct color to every category, using fallbacks as needed."""
    color_map, fallback_idx = {}, 0
    for cat in unique_cats:
        if cat in CATEGORY_COLORS:
            color_map[cat] = CATEGORY_COLORS[cat]
        else:
            color_map[cat] = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
    return color_map

def encode_full(name, cfg, texts, precomputed):
    """Feature matrix for an encoder string over the FULL dataset, standardized.
    Visualization only: the sparse block is fit on all texts at once."""
    emb, sparse = parse_encoder(name, cfg)
    blocks = []
    if emb is not None:
        blocks.append(precomputed[emb])
    if sparse is not None:
        vec, svd = make_vectorizer(sparse, cfg)
        X_sparse = vec.fit_transform(texts)
        safe = max(1, min(svd.n_components, min(X_sparse.shape) - 1))
        if safe != svd.n_components:
            svd.n_components = safe
        blocks.append(svd.fit_transform(X_sparse))
    if not blocks:
        raise ValueError(f"Unknown encoder: {name!r}")
    X = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
    return StandardScaler().fit_transform(X)

def compute_tsne(cfg, features):
    """Project a feature matrix to 2-D with t-SNE (cosine metric)."""
    perplexity = min(cfg.get('tsne_perplexity', 30), len(features) - 1)
    n_iter = cfg.get('tsne_n_iter', 1000)
    seed = cfg.get('random_seed', 42)

    # sklearn >= 1.5 renamed 'n_iter' to 'max_iter'.
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                    random_state=seed, metric='cosine',
                    learning_rate='auto', init='pca')
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                    random_state=seed, metric='cosine',
                    learning_rate='auto', init='pca')

    coords = tsne.fit_transform(features)
    return np.asarray(coords)

def style_tsne_axes(ax, title):
    ax.set_xlabel('t-SNE Dimension 1', labelpad=8)
    ax.set_ylabel('t-SNE Dimension 2', labelpad=8)
    ax.set_title(title, pad=16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.4, color='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

def plot_tsne(tsne_results, categories, unique_cats, color_map, save_dir, name,
              title='t-SNE Visualization of Component Feature Space'):
    """Combined t-SNE scatter, one color per category, legend on the right."""
    from matplotlib.gridspec import GridSpec
    setup_plot_style()

    n_cats = len(unique_cats)
    fig_height = max(8, n_cats * 0.42 + 2.0)
    legend_width = 3.4

    fig = plt.figure(figsize=(10 + legend_width, fig_height), dpi=300)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[10, legend_width], wspace=0.02)
    ax = fig.add_subplot(gs[0])
    lax = fig.add_subplot(gs[1])
    lax.set_axis_off()

    cat_arr = np.array(categories)
    handles = []
    for cat in unique_cats:
        mask = cat_arr == cat
        sc = ax.scatter(
            tsne_results[mask, 0], tsne_results[mask, 1],
            c=color_map[cat], label=cat, s=30, alpha=0.5,
            edgecolors='white', linewidth=0.6,
            zorder=1 if cat == 'Other' else 10, rasterized=True,
        )
        handles.append(sc)
    style_tsne_axes(ax, title)

    legend = lax.legend(
        handles=handles, labels=unique_cats, loc='center left',
        bbox_to_anchor=(0.0, 0.5), frameon=True, fancybox=False, shadow=False,
        fontsize=13, title='Component Type', title_fontsize=15,
        markerscale=1.4, handlelength=1.2, handleheight=1.2,
        borderpad=0.8, labelspacing=0.55, framealpha=0.95, edgecolor='#333333',
    )
    legend.get_title().set_fontweight('bold')

    save_figure(fig, save_dir, name, also_pdf=True)

def silhouette_by_category(cfg, features, categories, unique_cats):
    """Overall + per-category mean silhouette (cosine), subsampled if large."""
    categories = np.asarray(categories)
    n_points = len(features)
    seed = cfg.get('random_seed', 42)
    cap = cfg.get('max_silhouette_samples', 8000)

    overall_kwargs = {'metric': 'cosine'}
    if n_points > cap:
        overall_kwargs.update(sample_size=cap, random_state=seed)
    sil_overall = silhouette_score(features, categories, **overall_kwargs)

    if n_points > cap:
        rng = np.random.default_rng(seed)
        sel = rng.choice(n_points, size=cap, replace=False)
        feat_s, cat_s = features[sel], categories[sel]
    else:
        feat_s, cat_s = features, categories
    sil_samples = silhouette_samples(feat_s, cat_s, metric='cosine')

    rows = [{'Category': c,
             'Mean Silhouette': float(sil_samples[cat_s == c].mean())
                                if (cat_s == c).any() else np.nan}
            for c in unique_cats]
    table = (pd.DataFrame(rows)
             .sort_values('Mean Silhouette', ascending=False)
             .reset_index(drop=True))
    table = pd.concat([pd.DataFrame([{'Category': 'Overall',
                                      'Mean Silhouette': float(sil_overall)}]),
                       table], ignore_index=True)
    return sil_overall, table

def run_embedding_analysis(cfg, encoders, texts, raw_categories, precomputed):
    """t-SNE figure and silhouette table for every encoder's feature space."""
    if cfg.get('tsne_collapse_other', True):
        categories = ['Other' if c in OTHER_CATEGORIES else c
                      for c in raw_categories]
    else:
        categories = list(raw_categories)
    unique_cats = ordered_categories(categories)
    color_map = build_color_map(unique_cats)
    fig_dir = figures_dir(cfg)

    for enc in encoders:
        log(f"t-SNE for encoder '{enc}'")
        features = encode_full(enc, cfg, texts, precomputed)
        coords = compute_tsne(cfg, features)

        safe_enc = enc.replace('+', '_')
        plot_tsne(coords, categories, unique_cats, color_map, fig_dir,
                  f"tsne_{safe_enc}",
                  title=f't-SNE of {encoder_label(enc)} Feature Space')

        if cfg.get('silhouette_analysis', True):
            sil_overall, sil_table = silhouette_by_category(
                cfg, features, categories, unique_cats)
            log(f"Overall silhouette (cosine): {sil_overall:.3f}")
            if len(encoders) > 1:
                sil_table.insert(0, 'Encoder', encoder_label(enc))
                path = os.path.join(tables_dir(cfg),
                                    f"silhouette_by_category_{safe_enc}.csv")
                sil_table.to_csv(path, index=False)
                log(f"Saved silhouette_by_category_{safe_enc}.csv")
            else:
                save_table(sil_table, cfg, 'silhouette_by_category')

# =============================================================================
# RESULTS OUTPUT
# =============================================================================
def print_results_summary(df_metrics):
    """Aligned console table of mean +/- std for every combo, best R2 first."""
    banner("RESULTS SUMMARY (mean ± std across folds)")
    header = f"{'Combo':<34} {'R²':>16} {'MAE':>18} {'RMSE':>18} {'MAPE %':>19}"
    print(header)
    print('-' * len(header))
    for _, r in df_metrics.iterrows():
        print(f"{r['combo']:<34} "
              f"{r['R2_mean']:>7.4f} ±{r['R2_std']:>6.4f} "
              f"{r['MAE_mean']:>9.4f} ±{r['MAE_std']:>6.4f} "
              f"{r['RMSE_mean']:>9.4f} ±{r['RMSE_std']:>6.4f} "
              f"{r['MAPE_mean']:>10.2f} ±{r['MAPE_std']:>6.2f}")

def save_predictions(cfg, df_pred, combos, predict_acc):
    """One CSV with per-component fold mean/std for each combo + an ensemble."""
    if df_pred is None:
        return
    pred_df = df_pred.copy().reset_index(drop=True)
    mean_cols = []
    for (e, m) in combos:
        mean_p, std_p = aggregate_predictions(predict_acc[(e, m)])
        if mean_p is None:
            continue
        combo = combo_label(e, m)
        pred_df[f"{combo}_mean"] = mean_p
        pred_df[f"{combo}_std"] = std_p
        mean_cols.append(f"{combo}_mean")
    if mean_cols:
        stacked = pred_df[mean_cols].to_numpy(dtype=float)
        pred_df['Ensemble_mean'] = np.nanmean(stacked, axis=1)
        pred_df['Ensemble_std'] = np.nanstd(stacked, axis=1)
    save_table(pred_df, cfg, 'predictions')

# =============================================================================
# MAIN
# =============================================================================
def main(cfg=CONFIG):
    cfg = cfg.copy()
    cat_col, target = cfg.get('category_col', 'Component Category'), cfg['target_col']
    fig_dir = figures_dir(cfg)

    banner("MASS PREDICTION FROM COMPONENT DESCRIPTIONS")
    log(f"Device: {cfg['device']}  |  Seed: {cfg.get('random_seed', 42)}")
    log(f"Figures -> {fig_dir}")
    log(f"Tables  -> {tables_dir(cfg)}")
    set_seed(cfg.get('random_seed', 42))

    # --- 1. Data ---
    banner("1/6  DATA LOADING AND CLEANING")
    df, df_pred, df_before, df_after = load_data(cfg)

    # --- 2. Dataset summaries (on the full cleaned data) ---
    banner("2/6  DATASET SUMMARIES")
    save_table(dataset_summary(df_before, df_after, df, cfg), cfg,
               'dataset_summary')
    save_table(category_stats(df, cfg), cfg, 'dataset_by_category')

    sample_n = cfg.get('experiment_sample_size')
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n,
                       random_state=cfg.get('random_seed', 42)).reset_index(drop=True)

    # --- 3. Embeddings (computed once) ---
    banner("3/6  TEXT EMBEDDINGS")
    encoders = list(cfg.get('encoders', ['sbert']))
    models = list(cfg.get('models', ['mlp']))

    texts = df[cfg['text_col']].astype(str).tolist()
    y = df[target].values.astype(float)
    predict_texts = (df_pred[cfg['text_col']].astype(str).tolist()
                     if df_pred is not None else None)

    precomputed = {}
    registry = cfg.get('embedders', {})
    for key in embedders_used(encoders, cfg):
        spec = registry[key]
        prefix = spec.get('prefix', '')
        normalize = spec.get('normalize', False)
        log(f"Encoding with {key} ({spec['hf']})")
        model_st = SentenceTransformer(spec['hf'])
        if prefix:
            train_inputs = [prefix + t for t in texts]
        else:
            train_inputs = list(texts)
        train_emb = model_st.encode(train_inputs,
                                    show_progress_bar=cfg.get('verbose', True),
                                    convert_to_numpy=True,
                                    normalize_embeddings=normalize)
        precomputed[key] = np.asarray(train_emb)
        if predict_texts is not None:
            if prefix:
                pred_inputs = [prefix + t for t in predict_texts]
            else:
                pred_inputs = list(predict_texts)
            pred_emb = model_st.encode(pred_inputs,
                                       show_progress_bar=cfg.get('verbose', True),
                                       convert_to_numpy=True,
                                       normalize_embeddings=normalize)
            precomputed[key + '_predict'] = np.asarray(pred_emb)

    # --- 4. Optional Optuna tuning (MLP on SBERT features) ---
    nn_params = None
    if cfg.get('use_optuna', False) and 'mlp' in models and 'sbert' in precomputed:
        nn_params = tune_mlp(cfg, precomputed['sbert'], y)

    # --- 5. Cross-validated comparison grid ---
    n_folds = cfg.get('cv_folds', 10)
    all_combos = []
    for e in encoders:
        for m in models:
            all_combos.append((e, m))
    banner(f"4/6  {n_folds}-FOLD CROSS-VALIDATION "
           f"({len(all_combos)} encoder×model combinations)")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.get('random_seed', 42))
    combo_metrics = {c: [] for c in all_combos}
    predict_acc = {c: [] for c in all_combos}
    percat = {c: [] for c in all_combos}
    permass = {c: [] for c in all_combos}
    baseline_metrics = {b: [] for b in BASELINE_NAMES}
    last_fold, last_history = {}, {}
    _, mass_bin_names = mass_bin_labels(y, cfg.get('mass_bin_edges',
                                                   [0, 0.5, 1, 5, 10, 25]))

    for fold, (tr, te) in enumerate(kf.split(np.arange(len(y))), 1):
        fold_seed = cfg.get('random_seed', 42) + fold
        set_seed(fold_seed)
        log(f"Fold {fold}/{n_folds}  (train {len(tr)}, test {len(te)})")
        y_train, y_test = y[tr], y[te]
        train_texts = [texts[i] for i in tr]
        test_texts = [texts[i] for i in te]
        cat_test = df.iloc[te][cat_col].astype(str).values
        massbin_test, _ = mass_bin_labels(y_test, cfg.get('mass_bin_edges',
                                                          [0, 0.5, 1, 5, 10, 25]))

        for name, preds in baseline_predictions(y_train, df.iloc[tr],
                                                df.iloc[te], cfg).items():
            baseline_metrics[name].append(fold_metrics(fold, y_test, preds))

        for e in encoders:
            Xtr_raw, Xte_raw, Xpred_raw = encode_fold(
                e, cfg, train_texts, test_texts, precomputed=precomputed,
                tr_idx=tr, te_idx=te, predict_texts=predict_texts)
            scaler = StandardScaler().fit(Xtr_raw)
            Xtr = scaler.transform(Xtr_raw)
            Xte = scaler.transform(Xte_raw)
            if Xpred_raw is not None:
                Xpred = scaler.transform(Xpred_raw)
            else:
                Xpred = None

            for m in models:
                label = combo_label(e, m)
                try:
                    test_preds, ext_preds, history = fit_predict_combo(
                        m, cfg, Xtr, y_train, Xte, Xpred, nn_params, fold_seed)
                    combo_metrics[(e, m)].append(fold_metrics(fold, y_test, test_preds))
                    percat[(e, m)].extend(
                        per_group_metrics(fold, y_test, test_preds, cat_test))
                    permass[(e, m)].extend(
                        per_group_metrics(fold, y_test, test_preds, massbin_test))
                    if Xpred is not None and ext_preds is not None:
                        predict_acc[(e, m)].append(np.asarray(ext_preds))
                    if fold == n_folds:
                        last_fold[(e, m)] = (np.asarray(y_test), np.asarray(test_preds))
                        if history is not None:
                            last_history[(e, m)] = history
                except Exception as ex:
                    log(f"    {label:<32} FAILED: {ex}")
                    combo_metrics[(e, m)].append(
                        fold_metrics(fold, y_test, np.full(len(y_test), np.nan)))

    # --- 6. Aggregate, report, and plot ---
    banner("5/6  AGGREGATED RESULTS")
    result_rows = [build_combo_row(combo_label(e, m), e, m, combo_metrics[(e, m)])
                   for (e, m) in all_combos]
    result_rows += [build_combo_row(b, '-', b, baseline_metrics[b])
                    for b in BASELINE_NAMES]
    df_metrics = (pd.DataFrame(result_rows)
                  .sort_values('R2_mean', ascending=False)
                  .reset_index(drop=True))
    save_table(df_metrics, cfg, 'model_comparison')
    print_results_summary(df_metrics)

    banner("6/6  FIGURES AND PREDICTIONS")
    percat_agg = aggregate_pergroup(percat)
    if percat_agg:
        save_table(pergroup_table(percat_agg, all_combos, 'Component Category'),
                   cfg, 'error_by_category')

    permass_agg = aggregate_pergroup(permass)
    if permass_agg:
        save_table(pergroup_table(permass_agg, all_combos, 'Mass Range (g)',
                                  group_order=mass_bin_names),
                   cfg, 'error_by_mass_range')

    zoom_range = cfg.get('hexbin_zoom_range', (0.0, 1.0))
    for (e, m), (yt, yp) in last_fold.items():
        prefix = f"{e.replace('+', '_')}__{m}"
        save_hexbin_plot(yt, yp, fig_dir, prefix, target_col=target)
        save_hexbin_plot(yt, yp, fig_dir, prefix, target_col=target,
                         mass_range=zoom_range)
        save_residual_histogram(yt, yp, fig_dir, prefix)
    for (e, m), hist in last_history.items():
        save_learning_curve(hist, fig_dir, f"{e.replace('+', '_')}__{m}")

    if cfg.get('tsne_plot', True):
        run_embedding_analysis(cfg, encoders, texts,
                               df[cat_col].astype(str).tolist(), precomputed)

    save_predictions(cfg, df_pred, all_combos, predict_acc)

    log("Run complete.")
    return {'metrics_summary': df_metrics}

if __name__ == '__main__':
    main()
