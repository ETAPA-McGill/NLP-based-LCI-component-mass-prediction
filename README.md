# Predicting electronic component mass for life cycle assessment using 
# natural language processing: closing life cycle inventory data gaps 
# without manual feature engineering

Code for the paper *"Predicting electronic component mass for life cycle assessment using 
natural language processing: closing life cycle inventory data gaps 
without manual feature engineering"*
(McClay et al., in review).

The model predicts the unit mass of an electronic component directly from its
bill-of-materials (BoM) part description. Text is encoded with a sentence
transformer (SBERT, `all-mpnet-base-v2`) concatenated with character-level
TF-IDF features, and a small MLP regressor maps the combined representation to
mass. No manual feature extraction is required, which makes the approach
usable for the many component types whose descriptions are too inconsistent
for reliable feature engineering.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Inputs](#inputs)
- [What happens when you run it](#what-happens-when-you-run-it)
- [Outputs](#outputs)
- [Configuration reference](#configuration-reference)
- [Common customizations](#common-customizations)
- [Reproducing the paper results](#reproducing-the-paper-results)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

## Installation

```bash
git clone https://github.com/ETAPA-McGill/LCI_zero_feature_engineering.git
cd LCI_zero_feature_engineering
pip install -r requirements.txt
```

Python ≥ 3.9. A CUDA GPU is used automatically when available and speeds up
MLP training considerably; everything also runs on CPU. The first run
downloads the `all-mpnet-base-v2` sentence-transformer weights (~420 MB) from
Hugging Face and caches them locally.

## Quick start

1. Place your training CSV in `data/` (see [Inputs](#inputs)).
2. Open `mass_prediction.py` and edit the `CONFIG` dictionary at the top —
   at minimum, set `train_csv` to your file path.
3. Run:

```bash
python mass_prediction.py
```

Results appear in `figures/` and `tables/` inside `output_dir`
(the repository root by default).

For a fast sanity-check run before committing to the full experiment, set:

```python
'experiment_sample_size': 2000,   # subsample the training data
'cv_folds': 3,
'epochs': 30,
```

## Inputs

### Training CSV (`train_csv`, required)

One row per component. Required columns (names are configurable):

| Column               | CONFIG key     | Type   | Description                          |
| -------------------- | -------------- | ------ | ------------------------------------ |
| `Name`               | `text_col`     | string | The BoM part description text        |
| `Mass (g)`           | `target_col`   | float  | Reported unit mass in grams          |
| `Component Category` | `category_col` | string | Category label (used for baselines, error breakdowns, and the embedding-space analysis — **never** as a model input) |

Any other columns are ignored (columns whose name starts with `Unnamed` are
dropped when `drop_unnamed_columns` is on). Rows missing the text or the
target are dropped. Rows whose *normalized* text (stripped + lowercased) is
identical are merged into one, with the target averaged — this de-duplication
is what keeps the cross-validation honest, so leave it on unless you have a
specific reason not to.

### Prediction CSV (`predict_csv`, optional)

A list of components whose mass you want predicted. Only the text column is
required; every other column is carried through unchanged into
`predictions.csv`. Any prediction row whose text also appears in the training
data causes that training row to be **removed from training** before anything
is fit, so predictions are never made for memorized components. Set
`predict_csv` to `None` to skip this step entirely.

## What happens when you run it

The console walks through six numbered stages with timestamped progress lines:

```
1/6  DATA LOADING AND CLEANING      dedup counts, outlier-filter counts
2/6  DATASET SUMMARIES              writes dataset_summary.csv, dataset_by_category.csv
3/6  TEXT EMBEDDINGS                SBERT encoding progress bar (once, up front)
4/6  K-FOLD CROSS-VALIDATION        one line per fold; MLP folds show an epoch bar
5/6  AGGREGATED RESULTS             writes model_comparison.csv + prints a summary table
6/6  FIGURES AND PREDICTIONS        error tables, plots, t-SNE, predictions.csv
```

At the end of stage 5 you get an aligned console table of mean ± std for
every encoder × model combination and the four baselines, sorted by R²:

```
Combo                                   R²              MAE       ...
SBERT+CharTFIDF+MLP           0.63 ± 0.01     0.80 ± 0.02   ...
Baseline_CategoryMean         0.20 ± 0.01     1.61 ± 0.03   ...
```

**Runtime expectations.** With the full dataset (~63k rows) on a single
modern GPU, the default configuration (one encoder combo, MLP, 10 folds)
takes on the order of 1–2 hours, dominated by MLP training; the one-off SBERT
encoding takes a few minutes. On CPU expect several times longer. Each extra
encoder or model multiplies stage 4 accordingly. Enabling Optuna adds
`optuna_n_trials × optuna_cv_folds` additional MLP trainings before the main
run. t-SNE on the full dataset takes several minutes and needs a few GB of
RAM.

**Reproducibility.** All RNGs (Python, NumPy, PyTorch) are seeded from
`random_seed`, with a per-fold derived seed, so repeated runs on the same
data and configuration give the same folds and results (up to normal GPU
non-determinism in cuDNN kernels).

## Outputs

Two folders are created inside `output_dir`:

### `tables/`

**`model_comparison.csv`** — one row per encoder × model combination plus the
four baselines, sorted by R². Columns: `combo`, `encoder`, `model`, then
`R2_mean/R2_std`, `MAE_mean/MAE_std`, `RMSE_mean/RMSE_std`,
`MAPE_mean/MAPE_std` (mean and standard deviation across the CV folds; MAE
and RMSE in grams, MAPE in percent).

**`dataset_summary.csv`** — whole-dataset mass statistics at each cleaning
stage (after deduplication, after the outlier filter, final training set).
Columns: `Dataset`, `N`, `Mean (g)`, `Median (g)`, `Std Dev (g)`, `Max (g)`.

**`dataset_by_category.csv`** *(manuscript Table 4)* — the same statistics
for the final training set: a `Whole Dataset` row first, then one row per
component category in alphabetical order.

**`silhouette_by_category.csv`** *(manuscript Table 5)* — cosine silhouette
analysis of the encoder feature space. An `Overall` row first (the score
cited in the paper), then per-category mean silhouette sorted descending.
Columns: `Category`, `Mean Silhouette`. With several encoders configured, one
file per encoder is written instead (`silhouette_by_category_<encoder>.csv`).

**`error_by_mass_range.csv`** *(manuscript Table 6)* — held-out error binned
by reported mass. Columns: `Mass Range (g)` (e.g. `0.0 - 0.1`, with a final
open `> 25.0` bin), `Mean Count (n)` (average test components per fold in the
bin), `MAE`, `MAE_std`, `RMSE`, `RMSE_std` (grams; mean and std across
folds).

**`error_by_category.csv`** *(manuscript Table 7)* — the same breakdown by
component category, alphabetical. Same columns with `Component Category`
first. For both error tables, a leading `Model` column is added only when
more than one encoder × model combination was run.

**`predictions.csv`** — your prediction CSV with columns appended per
combination: `<Combo>_mean` (average prediction across the CV fold models,
in grams) and `<Combo>_std` (fold-to-fold spread — a rough per-component
uncertainty). When several combinations are run, `Ensemble_mean` and
`Ensemble_std` summarize the cross-model consensus.

### `figures/`

All figures are SVG (the t-SNE plot is also saved as PDF). File names embed
the encoder and model, e.g. `parity_residuals_sbert_tfidf_char__mlp.svg`.

| Figure                        | Contents                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `parity_residuals_*`          | Two hexbin panels from the last CV fold: (a) reported vs predicted mass with a 1:1 line, (b) residual (reported − predicted) vs reported mass. Shared log-scale color bar for counts. |
| `parity_residuals_*_zoom_*`   | The same panels rebuilt over only the `hexbin_zoom_range` window (default 0–1 g), so the sub-gram region is legible. |
| `residual_histogram_*`        | Residual distribution on a log frequency axis (last fold).                                     |
| `learning_curve_*`            | MLP training diagnostics from the last fold: training loss, validation R², validation MAE per epoch. |
| `tsne_<encoder>`              | 2-D t-SNE projection of the encoder feature space, one color per component category (minor categories folded into a grey `Other` when `tsne_collapse_other` is on). |

## Configuration reference

Everything is controlled from the `CONFIG` dictionary at the top of
`mass_prediction.py`.

### Input data

| Key                    | Default                | Meaning                                             |
| ---------------------- | ---------------------- | --------------------------------------------------- |
| `train_csv`            | —                      | Path to the training CSV                            |
| `predict_csv`          | `None`                 | Path to the prediction CSV, or `None` to skip       |
| `text_col`             | `'Name'`               | Column holding the part description                 |
| `target_col`           | `'Mass (g)'`           | Column holding the unit mass (grams)                |
| `category_col`         | `'Component Category'` | Column holding the category label                   |
| `drop_unnamed_columns` | `True`                 | Drop spreadsheet-artifact `Unnamed: n` columns      |
| `deduplicate`          | `True`                 | Merge rows with identical normalized text           |

### Outlier filtering

| Key                                    | Default         | Meaning                                              |
| -------------------------------------- | --------------- | ---------------------------------------------------- |
| `outlier_method`                       | `'mass_cutoff'` | `None`, `'iqr'`, `'percentile'`, or `'mass_cutoff'`  |
| `mass_cutoff_g`                        | `25.0`          | Upper mass cutoff in grams (`mass_cutoff` method)    |
| `iqr_multiplier`                       | `3.0`           | Tukey fence width (`iqr` method)                     |
| `percentile_lower` / `percentile_upper`| `0.0` / `99`    | Percentile bounds (`percentile` method)              |

### Encoders

| Key              | Default                 | Meaning                                                              |
| ---------------- | ----------------------- | -------------------------------------------------------------------- |
| `encoders`       | `['sbert+tfidf_char']`  | Any of `'sbert'`, `'tfidf'` (word 1–2-grams), `'tfidf_char'` (character 2–5-grams), or `+` combinations whose feature blocks are concatenated |
| `embedders`      | `{'sbert': …}`          | Registry of sentence-transformer models: Hugging Face name, optional input `prefix`, optional L2 `normalize` |
| `encoder_params` | see script              | `svd_components` (LSA dimensions the sparse features are reduced to, default 300), `tfidf_max_features`, `tfidf_ngram_range`, `char_ngram_range` |

Embeddings are computed once for the whole dataset (a fixed pretrained
transform, so this is leak-free); TF-IDF vocabularies, IDF weights, and the
SVD are refit on each training fold only.

### Models

| Key            | Default    | Meaning                                                                    |
| -------------- | ---------- | --------------------------------------------------------------------------- |
| `models`       | `['mlp']`  | Any of `'mlp'`, `'linear'`, `'ridge'`, `'random_forest'`, `'knn'`, `'hist_gbr'`; every encoder × model combination is evaluated |
| `model_params` | see script | Hyperparameters for the scikit-learn models (`ridge_alpha`, `rf_n_estimators`, `rf_max_depth`, `knn_n_neighbors`, `knn_weights`, `hgb_max_iter`, `hgb_learning_rate`) |

### MLP training

| Key                           | Default    | Meaning                                                       |
| ----------------------------- | ---------- | -------------------------------------------------------------- |
| `hidden_dim`                  | `512`      | Width of the first hidden layer (layer *i* has `hidden_dim / 2^i`) |
| `mlp_hidden_layers`           | `2`        | Number of hidden layers                                        |
| `dropout`                     | `0.1`      | Dropout after every hidden layer                               |
| `batch_size` / `epochs` / `lr`| `128` / `200` / `1.5e-4` | Standard AdamW training settings                 |
| `weight_decay`                | `0.03`     | AdamW weight decay                                             |
| `lr_scheduler`                | `'cosine'` | Cosine LR decay over `epochs`, or `None`                       |
| `grad_clip_norm`              | `1.0`      | Gradient-norm clipping, or `None`                              |
| `loss_function`               | `'huber'`  | `'mse'`, `'mae'`, or `'huber'` (with `huber_delta`, default 1.5) |
| `early_stopping_patience`     | `40`       | Stop after this many epochs without validation-R² improvement  |
| `early_stopping_val_fraction` | `0.15`     | Validation slice carved **from the training fold** for early stopping — the test fold is never touched during training |
| `target_normalization`        | `None`     | Train on a transformed target: `'log'`, `'log1p'`, `'cbrt'`, `'boxcox'`; predictions are always mapped back to grams before any metric |
| `device`                      | auto       | `'cuda'` if available, else `'cpu'`                            |

### Error analysis, embedding analysis, cross-validation

| Key                      | Default                              | Meaning                                          |
| ------------------------ | ------------------------------------ | ------------------------------------------------ |
| `mass_bin_edges`         | `[0, 0.1, 0.5, 1, 2, 5, 10, 25]`     | Bin edges (g) for the mass-range error table; a final open bin is added |
| `hexbin_zoom_range`      | `(0.0, 1.0)`                         | Window (g) for the zoomed hexbin figure          |
| `tsne_plot`              | `True`                               | Toggle the whole embedding-space analysis        |
| `tsne_collapse_other`    | `True`                               | Fold minor categories into `Other` for plotting/silhouette |
| `tsne_perplexity` / `tsne_n_iter` | `30` / `1000`               | t-SNE parameters                                 |
| `silhouette_analysis`    | `True`                               | Toggle the silhouette table                      |
| `max_silhouette_samples` | `8000`                               | Subsample cap for the silhouette computation     |
| `cv_folds`               | `10`                                 | Number of cross-validation folds                 |
| `random_seed`            | `42`                                 | Master seed for everything                       |
| `experiment_sample_size` | `None`                               | Randomly subsample the training data (quick runs) |

### Optuna hyperparameter search (MLP only)

| Key                   | Default | Meaning                                            |
| --------------------- | ------- | --------------------------------------------------- |
| `use_optuna`          | `False` | Tune the MLP before the main CV run (searches lr, weight decay, width, depth, dropout, Huber delta, target transform) |
| `optuna_n_trials`     | `50`    | Number of trials                                    |
| `optuna_timeout_s`    | `None`  | Optional wall-clock budget (seconds)                |
| `optuna_cv_folds`     | `3`     | Inner CV folds per trial                            |
| `optuna_target_norms` | all     | Target-transform candidates offered to the search   |

### Outputs

| Key                          | Default                    | Meaning                          |
| ---------------------------- | -------------------------- | --------------------------------- |
| `output_dir`                 | `'.'`                      | Root folder for all outputs       |
| `figures_dir` / `tables_dir` | `'figures'` / `'tables'`   | Subfolder names inside the root   |
| `verbose`                    | `True`                     | Progress bars for SBERT encoding  |

## Common customizations

**Compare several models at once.** Every encoder × model combination is
cross-validated and lands in `model_comparison.csv`:

```python
'encoders': ['sbert', 'tfidf_char', 'sbert+tfidf_char'],
'models':   ['mlp', 'linear', 'knn', 'hist_gbr'],
```

**Swap the sentence-transformer.** Register it and reference it in
`encoders`. Models that expect an input prefix (e.g. E5-style) can set one:

```python
'embedders': {
    'sbert':  {'hf': 'all-mpnet-base-v2', 'prefix': '', 'normalize': False},
    'minilm': {'hf': 'all-MiniLM-L6-v2',  'prefix': '', 'normalize': False},
},
'encoders': ['minilm+tfidf_char'],
```

**Train in log space.** `'target_normalization': 'log'` (or `'log1p'`,
`'cbrt'`, `'boxcox'`). All reported metrics stay in grams regardless, so
results remain directly comparable.

**Change the mass scope.** Raise/lower `mass_cutoff_g`, switch
`outlier_method`, or set it to `None` to keep everything.

**Tune the MLP.** `'use_optuna': True`. The best parameters found are used
for every fold of the main run. Expect the added runtime noted above.

**Predict for a new component list.** Keep the list in `predict_csv`; the
per-fold models trained during CV are reused for prediction, so no extra
training happens — `predictions.csv` simply appears at the end of the run.

## Reproducing the paper results

The committed defaults are the paper's reported configuration: deduplicated
data, 25 g cutoff, `sbert+tfidf_char` encoding (768-d SBERT + 300-d SVD of
character 2–5-gram TF-IDF), 2-hidden-layer MLP with Softplus head and Huber
loss, 10-fold CV, seed 42. Running `python mass_prediction.py` on the
released dataset regenerates the headline metrics (R² ≈ 0.63 for the combined
representation) and the appendix tables 4–7.

## Troubleshooting

- **`OSError`/download failure at stage 3** — the sentence-transformer
  weights could not be fetched; check internet access to `huggingface.co` or
  pre-download the model and point `embedders['sbert']['hf']` at the local
  path.
- **CUDA out of memory** — lower `batch_size` (e.g. 64) or `hidden_dim`, or
  run on CPU by setting `'device': 'cpu'`.
- **t-SNE killed / very slow** — it is the most memory-hungry step; set
  `'tsne_plot': False` for pure modeling runs, or subsample with
  `experiment_sample_size`.
- **Arial font warnings** — cosmetic; matplotlib falls back to DejaVu Sans
  when Arial is not installed.
- **`n_iter` deprecation from scikit-learn** — handled automatically; the
  script uses `max_iter` on scikit-learn ≥ 1.5 and falls back otherwise.

## Citation and license

If you use this code or data, please cite the paper (see `CITATION.cff`;
reference will be updated upon publication). Released under the MIT License.
Developed by the ETAPA research group at McGill University in collaboration
with Schneider Electric.
