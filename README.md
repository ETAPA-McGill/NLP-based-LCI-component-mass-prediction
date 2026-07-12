# Predicting Component Mass from BoM Text Descriptions — Without Manual Feature Engineering

Code for the paper *"Filling Life Cycle Inventory Gaps: Predicting Electronic
Component Mass from BoM Text Descriptions Without Manual Feature Engineering"*
(McClay et al., in review).

The model predicts the unit mass of an electronic component directly from its
bill-of-materials (BoM) part description. Text is encoded with a sentence
transformer (SBERT, `all-mpnet-base-v2`) concatenated with character-level
TF-IDF features, and a small MLP regressor maps the combined representation to
mass. No manual feature extraction is required, which makes the approach
usable for the many component types whose descriptions are too inconsistent
for reliable feature engineering.

## What the script does

`mass_prediction.py` runs the full experimental pipeline from the paper:

1. **Data cleaning** — drops incomplete rows, merges duplicate part
   descriptions (masses averaged), and applies a 25 g mass cutoff.
2. **Encoding** — SBERT embeddings (precomputed once) and/or TF-IDF features
   (fit on each training fold only, to avoid leakage).
3. **10-fold cross-validation** — every configured encoder × regressor
   combination is trained and evaluated on held-out folds, alongside four
   naive baselines (global/category mean and median).
4. **Error analysis** — metrics broken down by component category and by mass
   range, plus a t-SNE projection and silhouette analysis of the embedding
   space.
5. **Prediction** — optionally predicts masses for an unseen component list,
   with per-component fold means and standard deviations.

## Installation

```bash
git clone https://github.com/ETAPA-McGill/LCI_zero_feature_engineering.git
cd LCI_zero_feature_engineering
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically if available; the script also runs
on CPU (slower).

## Usage

Point the `CONFIG` block at the top of `mass_prediction.py` to your data,
then run:

```bash
python mass_prediction.py
```

The most relevant settings:

| Key                    | Default              | Meaning                                        |
| ---------------------- | -------------------- | ---------------------------------------------- |
| `train_csv`            | —                    | Training CSV (see `data/README.md`)            |
| `predict_csv`          | `None`               | Optional CSV of components to predict          |
| `encoders`             | `['sbert+tfidf_char']` | `'sbert'`, `'tfidf'`, `'tfidf_char'`, or `+` combos |
| `models`               | `['mlp']`            | `'mlp'`, `'linear'`, `'ridge'`, `'random_forest'`, `'knn'`, `'hist_gbr'` |
| `mass_cutoff_g`        | `25.0`               | Upper mass cutoff (grams)                      |
| `cv_folds`             | `10`                 | Cross-validation folds                         |
| `target_normalization` | `None`               | Optional target transform (`'log'`, `'log1p'`, …) |
| `use_optuna`           | `False`              | Hyperparameter search for the MLP              |
| `output_dir`           | `'.'`                | Where `figures/` and `tables/` are written     |

The defaults reproduce the configuration reported in the paper
(SBERT + character-level TF-IDF → MLP, 10-fold CV, seed 42).

## Outputs

All results are written to two folders inside `output_dir`:

**`tables/`**

| File                         | Contents (manuscript reference)                     |
| ---------------------------- | --------------------------------------------------- |
| `model_comparison.csv`       | R², MAE, RMSE, MAPE (mean ± std) per model and baseline |
| `dataset_summary.csv`        | Mass statistics at each cleaning stage              |
| `dataset_by_category.csv`    | Per-category mass statistics (Table 4)              |
| `silhouette_by_category.csv` | Embedding-space silhouette analysis (Table 5)       |
| `error_by_mass_range.csv`    | MAE/RMSE by mass range (Table 6)                    |
| `error_by_category.csv`      | MAE/RMSE by component category (Table 7)            |
| `predictions.csv`            | Per-component predictions for the prediction set    |

**`figures/`**

- `parity_residuals_*` — reported vs predicted hexbin and residual panels
  (full range and sub-gram zoom)
- `residual_histogram_*` — residual distribution (log frequency scale)
- `learning_curve_*` — MLP training diagnostics
- `tsne_*` — t-SNE projection of the feature space by component category

## Repository structure

```
├── mass_prediction.py   # full pipeline (config block at the top)
├── requirements.txt
├── data/                # place input CSVs here (see data/README.md)
├── CITATION.cff
└── LICENSE
```

## Citation

If you use this code or data, please cite the paper (reference to be updated
upon publication; see `CITATION.cff`).

## License

Released under the MIT License (see `LICENSE`).

## Acknowledgments

Developed by the ETAPA research group at McGill University in collaboration
with Schneider Electric.
