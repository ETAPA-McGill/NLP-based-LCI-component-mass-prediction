# SBERT-MLP Mass Regression

A sentence-embedding-based regression framework for predicting component mass from text descriptions. This repository accompanies the manuscript submitted to the *Journal of Industrial Ecology*.

---

## Overview

This tool trains a neural network regressor that takes free-text component descriptions (e.g. bill-of-materials entries) and predicts their unit mass. It works in two stages:

1. **Text encoding** — component names are converted to dense vector representations using a pretrained Sentence-BERT (SBERT) model.
2. **Regression** — a multi-layer perceptron (MLP) maps those embeddings to a positive mass estimate.

The pipeline includes K-fold cross-validation, multiple baseline comparisons, optional hyperparameter tuning via Optuna, and publication-ready diagnostic plots.

## Model Architecture

### Sentence-BERT Embedding

Text descriptions are encoded using the [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) model from the `sentence-transformers` library. This produces a 384-dimensional dense vector for each input string. The model is used as a frozen feature extractor — its weights are not fine-tuned during training.

Key properties of the embedding model:

- 12-layer MiniLM architecture distilled from a larger transformer
- Trained on over 1 billion sentence pairs for semantic similarity
- 384-dimensional output embeddings
- Suitable for short-to-medium text inputs (component names, part descriptions)

### MLP Regressor

The regression head is a three-layer feedforward network:

```
Input (384) → Linear → LeakyReLU → Dropout → BatchNorm
           → Linear → LeakyReLU → Dropout → BatchNorm
           → Linear → Softplus → Output (1)
```

Notable design choices:

- **Softplus activation** on the output layer constrains predictions to be strictly positive, which is physically appropriate for mass.
- **BatchNorm + Dropout** at each hidden layer for regularisation.
- **Huber loss** (default) for robustness to outliers in the training data, with a configurable delta parameter.
- **Early stopping** on validation R² to prevent overfitting.

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/<your-username>/sbert-mlp-mass-regression.git
cd sbert-mlp-mass-regression
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended but not required. The code automatically falls back to CPU if no GPU is detected.

## Input Data

### Training CSV

A CSV file with at minimum:

| Column | Description |
|---|---|
| `Name` | Free-text component description |
| `Mass (g)` | Known unit mass in grams |
| `Component Category` | Category label (used for baseline comparisons and summary statistics) |

The column names for `Name` and `Mass (g)` are configurable via the `text_col` and `target_col` settings.

### Prediction CSV (optional)

A CSV file with the same `Name` column, containing components for which mass predictions are desired. Set `predict_csv` to `'NONE'` if no prediction data is available — in that case, no duplicate removal between training and prediction sets is performed.

## Configuration

All settings are controlled through the `CONFIG` dictionary at the top of the script. Key parameters:

### Data and Filtering

| Parameter | Default | Description |
|---|---|---|
| `train_csv` | — | Path to training data CSV |
| `predict_csv` | `'NONE'` | Path to prediction CSV, or `'NONE'` to skip |
| `outlier_method` | `'percentile'` | Outlier removal strategy: `None`, `'iqr'`, `'percentile'`, or `'mass_cutoff'` |
| `percentile_lower` / `percentile_upper` | `0.0` / `95` | Bounds for percentile filtering |
| `mass_cutoff_g` | `25.0` | Hard upper mass limit in grams (inclusive; used with `'mass_cutoff'`) |

### Model and Training

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `'all-minilm-l12-v2'` | Sentence-BERT model identifier |
| `hidden_dim` | `256` | Hidden layer width |
| `dropout` | `0.1` | Dropout rate |
| `lr` | `3e-4` | Learning rate (AdamW) |
| `epochs` | `200` | Maximum training epochs |
| `early_stopping_patience` | `40` | Epochs without R² improvement before stopping |
| `loss_function` | `'huber'` | Loss function: `'mse'`, `'mae'`, or `'huber'` |
| `huber_delta` | `0.5` | Huber loss threshold |
| `cv_folds` | `10` | Number of cross-validation folds |
| `target_normalization` | `None` | Set to `'log'` for log₁₊ₓ target transform |

### Optuna Hyperparameter Tuning

| Parameter | Default | Description |
|---|---|---|
| `use_optuna` | `False` | Enable automated hyperparameter search |
| `optuna_n_trials` | `50` | Number of Optuna trials |

When enabled, Optuna tunes learning rate, hidden dimension, dropout, and batch size using 5-fold cross-validation with median pruning.

## Usage

1. Place your training CSV in the working directory.
2. Edit the `CONFIG` dictionary to match your file paths and desired settings.
3. Run the script:

```bash
python SBERT_MASTER_REG_BARE.py
```

## Outputs

### CSV Files

| File | Contents |
|---|---|
| `*_Acc_Metrics.csv` | Per-model R², MAE, and RMSE (mean ± std across folds) for the neural network and all baselines |
| `*_Predictions.csv` | Mean and standard deviation of predicted mass for each entry in the prediction set (across folds) |
| `*_Baseline_Values.csv` | Global and per-category mean/median mass from the training data |
| `*_Dataset_Summary.csv` | Dataset statistics at each cleaning stage (before/after outlier removal, final) |
| `*_Dataset_By_Category.csv` | Per-category count, mass distribution, skewness, and kurtosis |

### Diagnostic Plots (SVG)

| Plot | Description |
|---|---|
| `NN_hexbin_combined.svg` | Two-panel figure — (a) reported vs. predicted mass and (b) residuals vs. reported mass, both as hexbin density plots with a shared log-scale colorbar |
| `NN_residual_histogram.svg` | Histogram of residuals with log-scaled y-axis |
| `NN_learning_curve.svg` | Three-panel training curve — loss, validation R², and validation MAE per epoch |

All plots are styled for publication (Times New Roman, 1500 DPI export).

### Baseline Models

The pipeline automatically evaluates four baselines alongside the neural network for context:

- **Global Mean** — predicts the training set mean for all samples
- **Global Median** — predicts the training set median for all samples
- **Category Mean** — predicts the per-category training mean
- **Category Median** — predicts the per-category training median

## Reproducibility

All random seeds (Python, NumPy, PyTorch) are fixed via the `random_seed` parameter (default: 42). Feature scaling is performed within each cross-validation fold to prevent data leakage. When a prediction set is provided, overlapping text entries are removed from the training data before any model fitting.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
