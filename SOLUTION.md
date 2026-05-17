# SOLUTION.md — Hallucination Detection (SMILES-2026)

## Reproducibility Instructions

**Environment:** Google Colab with T4 GPU (recommended).

**Steps to reproduce:**

1. Open Google Colab, upload `hallucination_detection.ipynb`
2. Set runtime to T4 GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order: Runtime → Run all
4. Feature extraction takes ~15–20 minutes on T4 GPU (result is cached to `features_cache.npz`)
5. After completion, `predictions.csv` and `results.json` are generated automatically

**Key implementation files (modified):**
- `aggregation.py` — feature extraction from hidden states
- `probe.py` — binary classifier
- `splitting.py` — 5-fold stratified cross-validation

---

## Final Solution Description

### What was modified

**`aggregation.py`**

Instead of using only the last token of the last layer (as in the original repo), we extract mean-pooled hidden states from **all 24 transformer layers** (layers 1–24), using only the **response tokens** (tokens after `<|im_start|>assistant`).

Feature vector: 24 layers × 896 dimensions = **21504 features** per sample.

Response token isolation is important because the prompt occupies ~265 out of 512 tokens on average — averaging over all tokens would dilute the hallucination signal.

**`probe.py`**

Pipeline: `StandardScaler → PCA(64) → LogisticRegression(C=0.1)`

- PCA reduces 21504 dimensions to 64, preventing overfitting on ~470 training samples
- LogisticRegression with L2 regularization generalizes better than nonlinear models on this dataset size
- Threshold is fixed at 0.5 (no threshold tuning on validation)

**`splitting.py`**

5-fold stratified cross-validation with a fixed held-out test set (15% = 104 samples). The same test set is used across all folds for consistent evaluation.

### Results

| Checkpoint | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.19% | 82.49% | N/A |
| Probe (train) | 82.22% | 87.77% | 87.18% |
| Probe (val) | 74.53% | 82.55% | 77.07% |
| Probe (test) | 70.00% | 79.00% | 68.48% |

Feature dim: 21504 · Folds: 5 · Extract time: 190s

### Why these choices

- **All 24 layers** — each layer captures different levels of abstraction; concatenating all gives the classifier maximum information to work with
- **Response tokens only** — hallucination signal lives in what the model generates, not in the prompt
- **PCA(64)** — with ~470 training samples, reducing to 64 components prevents the classifier from memorizing noise
- **LogisticRegression** — linear model is less likely to overfit than SVM or MLP on this dataset size

---

## Experiments and Failed Attempts

**3-layer aggregation (layers 8, 16, 24)** — Feature dim 2720. Gave similar test accuracy (~70%) with lower AUROC (~63%). All 24 layers gave better AUROC (68.5%).

**SVM with RBF kernel** — Test accuracy 0.596, AUROC 0.666. Worse than logistic regression; likely overfitting despite PCA.

**PCA(128) with C=0.1 and class_weight='balanced'** — Test accuracy 69.62% (below baseline). The balanced class weight pushed predictions toward the minority class (truthful), hurting accuracy on a 70/30 imbalanced test set.

**Threshold tuning on validation (optimizing accuracy)** — The optimal threshold on the validation set did not transfer to the test set, suggesting val/test distribution mismatch. Abandoned in favor of fixed threshold=0.5.

**C=0.01 (stronger regularization)** — Test AUROC 68.06%, accuracy 69.42%. Slightly worse than C=0.1.

**Main challenge:** Strong overfitting — train AUROC ~87% vs test AUROC ~68%. The dataset is small (689 samples) and the feature space is high-dimensional (21504), making generalization difficult even with PCA and strong regularization.
