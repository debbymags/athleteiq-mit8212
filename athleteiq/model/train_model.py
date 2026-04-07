"""
AthleteIQ — Athlete Injury Risk Prediction Model (Rigorous Version)
====================================================================
MIT 8212 Seminar: AI-Enabled Predictive Analytics in Sports
Miva Open University, 2025/2026

Model      : Gradient Boosted Trees (GradientBoostingClassifier, scikit-learn)
Dataset    : Synthetic, generated from published feature distributions
            

Improvements over v1
---------------------
  1. Nested cross-validation — outer CV for unbiased generalisation estimate,
     inner CV for hyperparameter search (GridSearchCV). Prevents leakage.
  2. Expanded metrics — AUC-ROC, AUC-PR, F1, Precision, Recall, MCC,
     Brier Score, Balanced Accuracy, per-fold variability.
  3. Calibrated probabilities — CalibratedClassifierCV (isotonic) so that
     output probabilities are reliable and suitable for the risk score.
  4. Threshold optimisation on held-out validation folds only — no leakage.
  5. Baseline comparison — Logistic Regression and Dummy classifier included
     to show the GBT provides genuine lift.
  6. Class imbalance handling — class_weight='balanced' noted; balanced
     accuracy and MCC used as primary metrics (robust to imbalance).
  7. Two feature importance methods — impurity-based (fast) and permutation
     importance (model-agnostic, unbiased) on the held-out test set.
  8. Full JSON export of all metrics for report and dashboard use.

Usage
-----
  python train_model.py

Requirements
------------
  pip install scikit-learn pandas numpy joblib
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import json
import joblib
import os
import warnings
from collections import Counter

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

N            = 2000   # total samples
OUTER_FOLDS  = 5      # folds for outer CV  (unbiased generalisation estimate)
INNER_FOLDS  = 3      # folds for inner CV  (hyperparameter search)
TEST_SIZE    = 0.20   # held-out test set fraction
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "weekly_load_mins",
    "fatigue_score",
    "sleep_quality",
    "muscle_soreness",
    "prior_injuries",
    "days_no_rest",
    "age",
    "position_risk",
]

# ─── SECTION 1: DATA GENERATION ──────────────────────────────────────────────
# Distributions grounded in:
#   Rossi et al. (2022), Claudino et al. (2023), van Dyk et al. (2024)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  AthleteIQ  |  Injury Risk Model  |  Rigorous Training Pipeline")
print("=" * 65)
print("\n[1/7]  Generating synthetic dataset ...")

weekly_load     = np.clip(np.random.normal(250, 80, N), 60, 600)
fatigue         = np.clip(np.random.normal(5.0, 2.0, N), 0, 10)
sleep_quality   = np.clip(np.random.normal(6.5, 1.8, N), 0, 10)
muscle_soreness = np.clip(np.random.normal(4.5, 2.2, N), 0, 10)
prior_injuries  = np.random.choice([0,1,2,3,4], N, p=[0.50,0.28,0.13,0.06,0.03])
days_no_rest    = np.clip(np.random.poisson(3.5, N), 0, 14)
age             = np.clip(np.random.normal(25, 4, N), 16, 40)
position_risk   = np.random.choice([0.5,0.7,0.8,0.9], N, p=[0.08,0.32,0.35,0.25])

# Label generation: logistic function with literature-grounded coefficients
log_odds = (
    -4.5
    + 0.008 * weekly_load
    + 0.28  * fatigue
    - 0.22  * sleep_quality
    + 0.18  * muscle_soreness
    + 0.45  * prior_injuries
    + 0.20  * days_no_rest
    + 0.06  * (age - 16)
    + 2.10  * position_risk
    + np.random.normal(0, 0.5, N)
)
prob_injury  = 1 / (1 + np.exp(-log_odds))
injury_label = (np.random.uniform(0, 1, N) < prob_injury).astype(int)

df = pd.DataFrame({
    "weekly_load_mins" : weekly_load,
    "fatigue_score"    : fatigue,
    "sleep_quality"    : sleep_quality,
    "muscle_soreness"  : muscle_soreness,
    "prior_injuries"   : prior_injuries.astype(float),
    "days_no_rest"     : days_no_rest.astype(float),
    "age"              : age,
    "position_risk"    : position_risk,
    "injury_risk"      : injury_label,
})

X = df[FEATURE_COLS].values
y = df["injury_risk"].values

n_pos = int(y.sum())
n_neg = int((y == 0).sum())
imbalance_ratio = n_pos / n_neg

print(f"       Samples      : {N}")
print(f"       Injury rate  : {y.mean():.1%}  (pos={n_pos}, neg={n_neg})")
print(f"       Class ratio  : {imbalance_ratio:.2f}:1")
print(f"       NOTE: Balanced Accuracy and MCC are primary metrics")
print(f"             as they are robust to class imbalance.")

# ─── SECTION 2: TRAIN / TEST SPLIT ───────────────────────────────────────────
# Stratified split preserves class ratio in both sets.
# The test set is held out completely — touched only at final evaluation.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[2/7]  Splitting data  (train 80% / test 20%, stratified) ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)
print(f"       Train : {len(X_train)} samples")
print(f"       Test  : {len(X_test)} samples")

# ─── SECTION 3: NESTED CROSS-VALIDATION ──────────────────────────────────────
#
# Design rationale
# ────────────────
# Nested CV is the gold standard for small-to-medium datasets.
# It prevents the optimistic bias that occurs when the same data is used
# both to select hyperparameters and to estimate generalisation performance.
#
# Structure:
#   Outer loop  (k=5, StratifiedKFold):
#       Each iteration uses 4/5 of training data to tune + train,
#       and 1/5 to evaluate. Yields 5 unbiased performance estimates.
#
#   Inner loop  (k=3, StratifiedKFold inside GridSearchCV):
#       Hyperparameters are searched using only the outer training portion.
#       The outer validation fold is never seen during tuning.
#
# Per-fold threshold optimisation:
#   The decision threshold (converting probabilities to labels) is chosen
#   to maximise F1 on each fold's own validation data — independently.
#   This avoids leaking threshold information across folds.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[3/7]  Nested cross-validation  "
      f"(outer={OUTER_FOLDS}-fold, inner GridSearch {INNER_FOLDS}-fold) ...")
print(f"       Searching hyperparameter grid — this may take ~60 seconds ...")

param_grid = {
    "n_estimators"    : [100, 200],
    "max_depth"       : [3, 4],
    "learning_rate"   : [0.05, 0.10],
    "subsample"       : [0.8, 1.0],
    "min_samples_leaf": [10, 20],
}

outer_cv   = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=fold_idx)

    # Inner: hyperparameter search
    grid_search = GridSearchCV(
        estimator  = GradientBoostingClassifier(random_state=42),
        param_grid = param_grid,
        cv         = inner_cv,
        scoring    = "roc_auc",
        n_jobs     = -1,
        refit      = True,
    )
    grid_search.fit(X_tr, y_tr)

    # Calibrate on the outer training split (inner CV handles avoiding leakage)
    calibrated = CalibratedClassifierCV(
        estimator = GradientBoostingClassifier(
            **grid_search.best_params_, random_state=42
        ),
        method = "isotonic",
        cv     = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=fold_idx),
    )
    calibrated.fit(X_tr, y_tr)

    y_prob_val = calibrated.predict_proba(X_val)[:, 1]

    # Threshold optimisation: maximise F1 on THIS fold's validation data only
    thresholds  = np.arange(0.20, 0.80, 0.01)
    fold_f1s    = [
        f1_score(y_val, (y_prob_val >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_thresh = float(thresholds[np.argmax(fold_f1s)])
    y_pred_val  = (y_prob_val >= best_thresh).astype(int)

    fold_results.append({
        "fold"              : fold_idx + 1,
        "best_params"       : grid_search.best_params_,
        "best_thresh"       : best_thresh,
        "auc_roc"           : roc_auc_score(y_val, y_prob_val),
        "auc_pr"            : average_precision_score(y_val, y_prob_val),
        "f1"                : f1_score(y_val, y_pred_val, zero_division=0),
        "precision"         : precision_score(y_val, y_pred_val, zero_division=0),
        "recall"            : recall_score(y_val, y_pred_val, zero_division=0),
        "balanced_accuracy" : balanced_accuracy_score(y_val, y_pred_val),
        "mcc"               : matthews_corrcoef(y_val, y_pred_val),
        "brier_score"       : brier_score_loss(y_val, y_prob_val),
        "calibrated_model"  : calibrated,
    })

    print(f"       Fold {fold_idx+1}: "
          f"AUC-ROC={fold_results[-1]['auc_roc']:.4f}  "
          f"AUC-PR={fold_results[-1]['auc_pr']:.4f}  "
          f"F1={fold_results[-1]['f1']:.4f}  "
          f"MCC={fold_results[-1]['mcc']:.4f}  "
          f"Bal-Acc={fold_results[-1]['balanced_accuracy']:.4f}  "
          f"Thresh={best_thresh:.2f}")

# ─── SECTION 4: AGGREGATE NESTED CV RESULTS ──────────────────────────────────

print(f"\n[4/7]  Aggregating cross-validation results ...")

metric_keys = [
    "auc_roc", "auc_pr", "f1", "precision", "recall",
    "balanced_accuracy", "mcc", "brier_score", "best_thresh"
]

cv_summary = {}
for key in metric_keys:
    vals = np.array([r[key] for r in fold_results])
    cv_summary[key] = {
        "mean"    : float(np.mean(vals)),
        "std"     : float(np.std(vals)),
        "min"     : float(np.min(vals)),
        "max"     : float(np.max(vals)),
        "per_fold": vals.tolist(),
    }

label_map = {
    "auc_roc"          : "AUC-ROC",
    "auc_pr"           : "AUC-PR  (Avg Precision)",
    "f1"               : "F1 Score",
    "precision"        : "Precision",
    "recall"           : "Recall  (Sensitivity)",
    "balanced_accuracy": "Balanced Accuracy",
    "mcc"              : "Matthews Corr Coef (MCC)",
    "brier_score"      : "Brier Score  (lower=better)",
    "best_thresh"      : "Optimal Threshold",
}

print(f"\n  Nested CV Summary  ({OUTER_FOLDS}-fold outer, {INNER_FOLDS}-fold inner GridSearch)")
print(f"  {'Metric':<28}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print(f"  {'-'*66}")
for key in metric_keys:
    v = cv_summary[key]
    print(f"  {label_map[key]:<28}  {v['mean']:>8.4f}  {v['std']:>8.4f}"
          f"  {v['min']:>8.4f}  {v['max']:>8.4f}")

# Consensus hyperparameters (majority vote across outer folds)
param_votes = Counter(
    tuple(sorted(r["best_params"].items())) for r in fold_results
)
best_params_final = dict(param_votes.most_common(1)[0][0])
mean_thresh = cv_summary["best_thresh"]["mean"]

print(f"\n  Consensus best hyperparameters (majority vote):")
for k, v in best_params_final.items():
    print(f"       {k:<22} = {v}")
print(f"  Mean optimal threshold: {mean_thresh:.3f}")

# ─── SECTION 5: FINAL MODEL ───────────────────────────────────────────────────
# Retrain on the full training set using consensus hyperparameters.
# Calibrate with isotonic regression via cross-validated wrapper.
# Evaluate ONCE on the held-out test set — no further adjustment.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[5/7]  Training final model on full training set ...")

final_model = CalibratedClassifierCV(
    estimator = GradientBoostingClassifier(**best_params_final, random_state=42),
    method    = "isotonic",
    cv        = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42),
)
final_model.fit(X_train, y_train)

# ── Single evaluation on held-out test set ──
y_prob_test = final_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= mean_thresh).astype(int)

test_auc_roc   = roc_auc_score(y_test, y_prob_test)
test_auc_pr    = average_precision_score(y_test, y_prob_test)
test_f1        = f1_score(y_test, y_pred_test, zero_division=0)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)
test_recall    = recall_score(y_test, y_pred_test, zero_division=0)
test_bal_acc   = balanced_accuracy_score(y_test, y_pred_test)
test_mcc       = matthews_corrcoef(y_test, y_pred_test)
test_brier     = brier_score_loss(y_test, y_prob_test)
test_accuracy  = accuracy_score(y_test, y_pred_test)
cm             = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
npv            = tn / (tn + fn) if (tn + fn) > 0 else 0.0

# Calibration quality
prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)
calibration_gap = float(np.mean(np.abs(prob_true - prob_pred)))

print(f"\n  Final Model — Hold-Out Test Set  (threshold = {mean_thresh:.3f})")
print(f"  {'Metric':<32}  {'Value':>10}")
print(f"  {'-'*45}")
print(f"  {'AUC-ROC':<32}  {test_auc_roc:>10.4f}")
print(f"  {'AUC-PR  (Avg Precision)':<32}  {test_auc_pr:>10.4f}")
print(f"  {'F1 Score':<32}  {test_f1:>10.4f}")
print(f"  {'Precision  (PPV)':<32}  {test_precision:>10.4f}")
print(f"  {'Recall  (Sensitivity)':<32}  {test_recall:>10.4f}")
print(f"  {'Specificity':<32}  {specificity:>10.4f}")
print(f"  {'Neg Predictive Value (NPV)':<32}  {npv:>10.4f}")
print(f"  {'Balanced Accuracy':<32}  {test_bal_acc:>10.4f}")
print(f"  {'Matthews Corr Coef (MCC)':<32}  {test_mcc:>10.4f}")
print(f"  {'Brier Score  (lower=better)':<32}  {test_brier:>10.4f}")
print(f"  {'Calibration Gap (|true-pred|)':<32}  {calibration_gap:>10.4f}")
print(f"  {'Raw Accuracy':<32}  {test_accuracy:>10.4f}")
print(f"\n  Confusion Matrix  (threshold = {mean_thresh:.3f}):")
print(f"                   Pred No Injury  Pred Injury")
print(f"  Actual No Injury  TN={tn:<6}       FP={fp}")
print(f"  Actual Injury     FN={fn:<6}       TP={tp}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_test,
                             target_names=["No Injury", "Injury"], digits=4))

# ─── SECTION 6: BASELINE COMPARISON ──────────────────────────────────────────
# Comparing against Logistic Regression and a majority-class Dummy classifier
# shows whether the GBT provides genuine predictive lift beyond simple baselines.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[6/7]  Baseline model comparison  ({OUTER_FOLDS}-fold CV) ...")

baselines = {
    "Majority Class (Dummy)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression"   : Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ]),
}

outer_cv_b = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
baseline_results = {}

for name, clf in baselines.items():
    scores = cross_validate(
        clf, X_train, y_train,
        cv      = outer_cv_b,
        scoring = {
            "auc_roc": "roc_auc",
            "auc_pr" : "average_precision",
            "f1"     : "f1",
            "bal_acc": "balanced_accuracy",
        },
        return_train_score=False,
    )
    baseline_results[name] = {
        "auc_roc_mean" : float(np.mean(scores["test_auc_roc"])),
        "auc_roc_std"  : float(np.std(scores["test_auc_roc"])),
        "auc_pr_mean"  : float(np.mean(scores["test_auc_pr"])),
        "f1_mean"      : float(np.mean(scores["test_f1"])),
        "bal_acc_mean" : float(np.mean(scores["test_bal_acc"])),
    }

print(f"\n  {'Model':<30}  {'AUC-ROC':>14}  {'AUC-PR':>10}  {'F1':>8}  {'Bal-Acc':>10}")
print(f"  {'-'*78}")
for name, res in baseline_results.items():
    print(f"  {name:<30}  "
          f"{res['auc_roc_mean']:.4f} ± {res['auc_roc_std']:.4f}  "
          f"{res['auc_pr_mean']:.4f}      "
          f"{res['f1_mean']:.4f}    "
          f"{res['bal_acc_mean']:.4f}")

print(f"  {'GBT  (Nested CV)':<30}  "
      f"{cv_summary['auc_roc']['mean']:.4f} ± {cv_summary['auc_roc']['std']:.4f}  "
      f"{cv_summary['auc_pr']['mean']:.4f}      "
      f"{cv_summary['f1']['mean']:.4f}    "
      f"{cv_summary['balanced_accuracy']['mean']:.4f}")
print(f"  {'GBT  (Test Set — final)':<30}  "
      f"{test_auc_roc:.4f}             "
      f"{test_auc_pr:.4f}      "
      f"{test_f1:.4f}    "
      f"{test_bal_acc:.4f}")

lr_auc = baseline_results["Logistic Regression"]["auc_roc_mean"]
print(f"\n  GBT lift over Logistic Regression: "
      f"AUC-ROC +{test_auc_roc - lr_auc:.4f}")

# ─── SECTION 7: FEATURE IMPORTANCE ───────────────────────────────────────────
# Two methods used for robustness:
#
# (a) Impurity-based importance (built-in GBT)
#     Fast. Can be biased toward high-cardinality or continuous features.
#     Extracted from a plain GBT retrained on X_train (the calibrated
#     wrapper does not expose feature_importances_ directly).
#
# (b) Permutation importance (model-agnostic)
#     Each feature is randomly shuffled 30 times on the held-out test set.
#     The mean drop in AUC-ROC is recorded. Unbiased and directly reflects
#     impact on out-of-sample predictive performance.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[7/7]  Feature importance analysis ...")

# (a) Impurity importance — retrain plain GBT for access to attribute
plain_gbt = GradientBoostingClassifier(**best_params_final, random_state=42)
plain_gbt.fit(X_train, y_train)
impurity_imp = plain_gbt.feature_importances_

# (b) Permutation importance — 30 repeats on test set
perm = permutation_importance(
    final_model, X_test, y_test,
    n_repeats    = 30,
    random_state = 42,
    scoring      = "roc_auc",
    n_jobs       = -1,
)
perm_means = perm.importances_mean
perm_stds  = perm.importances_std

sort_idx = np.argsort(perm_means)[::-1]

print(f"\n  {'Feature':<22}  {'Perm Imp (AUC drop)':<26}  {'Impurity Imp':>14}  Bar")
print(f"  {'-'*80}")
for i in sort_idx:
    bar = "█" * max(0, int(perm_means[i] * 400))
    print(f"  {FEATURE_COLS[i]:<22}  "
          f"{perm_means[i]:+.4f} ± {perm_stds[i]:.4f}           "
          f"{impurity_imp[i]:.4f}        {bar}")

# Normalised impurity weights for JS export
total_imp = impurity_imp.sum()
js_weights = {
    feat: round(float(imp / total_imp), 6)
    for feat, imp in zip(FEATURE_COLS, impurity_imp)
}

# Probability range anchors for dashboard risk score mapping
p5  = float(np.percentile(y_prob_test, 5))
p95 = float(np.percentile(y_prob_test, 95))
print(f"\n  Calibration gap (mean |true - pred| on test set): {calibration_gap:.4f}")
print(f"  Probability range (p5–p95 on test set): {p5:.4f} – {p95:.4f}")

# ─── SAVE ALL OUTPUTS ─────────────────────────────────────────────────────────

print(f"\n{'─'*65}  Saving outputs ...")

# 1. Trained model
joblib.dump(final_model, os.path.join(OUTPUT_DIR, "athlete_iq_model.pkl"))

# 2. Feature importances (both methods)
feat_imp_export = {
    feat: {
        "impurity_importance"    : round(float(impurity_imp[i]), 6),
        "permutation_importance" : round(float(perm_means[i]), 6),
        "permutation_std"        : round(float(perm_stds[i]), 6),
    }
    for i, feat in enumerate(FEATURE_COLS)
}
with open(os.path.join(OUTPUT_DIR, "feature_importances.json"), "w") as f:
    json.dump(feat_imp_export, f, indent=2)

# 3. Full metrics export
metrics_export = {
    "model"        : "GradientBoostingClassifier + CalibratedClassifierCV (isotonic)",
    "best_params"  : best_params_final,
    "nested_cv"    : {
        "outer_folds" : OUTER_FOLDS,
        "inner_folds" : INNER_FOLDS,
        "per_fold"    : [
            {k: v for k, v in r.items() if k != "calibrated_model"}
            for r in fold_results
        ],
        "summary"     : {
            k: {sk: (round(sv, 6) if isinstance(sv, float) else sv)
                for sk, sv in v.items()}
            for k, v in cv_summary.items()
        },
    },
    "test_set"     : {
        "n_samples"         : int(len(y_test)),
        "threshold"         : round(mean_thresh, 4),
        "auc_roc"           : round(test_auc_roc, 4),
        "auc_pr"            : round(test_auc_pr, 4),
        "f1"                : round(test_f1, 4),
        "precision"         : round(test_precision, 4),
        "recall"            : round(test_recall, 4),
        "specificity"       : round(specificity, 4),
        "npv"               : round(npv, 4),
        "balanced_accuracy" : round(test_bal_acc, 4),
        "mcc"               : round(test_mcc, 4),
        "brier_score"       : round(test_brier, 4),
        "accuracy"          : round(test_accuracy, 4),
        "calibration_gap"   : round(calibration_gap, 4),
        "confusion_matrix"  : {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
    },
    "baselines"    : baseline_results,
    "dataset"      : {
        "n_total"         : N,
        "n_train"         : int(len(X_train)),
        "n_test"          : int(len(X_test)),
        "injury_rate"     : round(float(y.mean()), 4),
        "imbalance_ratio" : round(float(imbalance_ratio), 4),
        "source"          : "Synthetic — generated from published feature distributions",
        "references"      : [
            "Rossi et al. (2022). PLOS ONE 17(1), e0262379.",
            "Claudino et al. (2023). Sports Medicine Open 9(1), 44.",
            "van Dyk et al. (2024). BJSM 58(2), 67-72.",
        ],
    },
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics_export, f, indent=2)

# 4. JS-ready weights for dashboard
js_export = {
    "model_type"        : "GradientBoostingClassifier + Isotonic Calibration",
    "best_params"       : best_params_final,
    "auc_roc_cv"        : round(cv_summary["auc_roc"]["mean"], 4),
    "auc_roc_test"      : round(test_auc_roc, 4),
    "auc_pr_cv"         : round(cv_summary["auc_pr"]["mean"], 4),
    "f1_cv"             : round(cv_summary["f1"]["mean"], 4),
    "mcc_cv"            : round(cv_summary["mcc"]["mean"], 4),
    "balanced_acc_cv"   : round(cv_summary["balanced_accuracy"]["mean"], 4),
    "brier_score_test"  : round(test_brier, 4),
    "optimal_threshold" : round(mean_thresh, 4),
    "prob_p5"           : round(p5, 4),
    "prob_p95"          : round(p95, 4),
    "feature_weights"   : js_weights,
    "feature_order"     : FEATURE_COLS,
    "dataset_size"      : N,
    "injury_rate"       : round(float(y.mean()), 4),
    "framework"         : "scikit-learn 1.8.0 · Python 3.12",
    "note"              : (
        "Nested 5-fold outer / 3-fold inner GridSearchCV. "
        "Probabilities calibrated with isotonic regression. "
        "Threshold tuned per fold on validation data (no leakage). "
        "See train_model.py for full pipeline documentation."
    ),
}
with open(os.path.join(OUTPUT_DIR, "js_weights.json"), "w") as f:
    json.dump(js_export, f, indent=2)

# 5. Dataset sample (first 200 rows)
df.head(200).to_csv(os.path.join(OUTPUT_DIR, "dataset_sample.csv"), index=False)

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  TRAINING COMPLETE")
print(f"{'='*65}")
print(f"  Nested CV  AUC-ROC       : {cv_summary['auc_roc']['mean']:.4f} ± {cv_summary['auc_roc']['std']:.4f}")
print(f"  Nested CV  AUC-PR        : {cv_summary['auc_pr']['mean']:.4f} ± {cv_summary['auc_pr']['std']:.4f}")
print(f"  Nested CV  F1            : {cv_summary['f1']['mean']:.4f} ± {cv_summary['f1']['std']:.4f}")
print(f"  Nested CV  MCC           : {cv_summary['mcc']['mean']:.4f} ± {cv_summary['mcc']['std']:.4f}")
print(f"  Nested CV  Bal. Accuracy : {cv_summary['balanced_accuracy']['mean']:.4f} ± {cv_summary['balanced_accuracy']['std']:.4f}")
print(f"  Nested CV  Brier Score   : {cv_summary['brier_score']['mean']:.4f} ± {cv_summary['brier_score']['std']:.4f}")
print(f"  ─────────────────────────────────────────────────")
print(f"  Test Set   AUC-ROC       : {test_auc_roc:.4f}")
print(f"  Test Set   AUC-PR        : {test_auc_pr:.4f}")
print(f"  Test Set   F1            : {test_f1:.4f}")
print(f"  Test Set   MCC           : {test_mcc:.4f}")
print(f"  Test Set   Bal. Accuracy : {test_bal_acc:.4f}")
print(f"  Test Set   Brier Score   : {test_brier:.4f}")
print(f"  Test Set   Calib. Gap    : {calibration_gap:.4f}")
print(f"  ─────────────────────────────────────────────────")
lr_auc = baseline_results["Logistic Regression"]["auc_roc_mean"]
lr_pr  = baseline_results["Logistic Regression"]["auc_pr_mean"]
print(f"  GBT lift vs. Logistic Regression:")
print(f"    AUC-ROC  +{test_auc_roc - lr_auc:.4f}")
print(f"    AUC-PR   +{test_auc_pr  - lr_pr:.4f}")
print(f"{'='*65}")
print(f"\n  Outputs: {OUTPUT_DIR}/")
print(f"    athlete_iq_model.pkl      (serialised calibrated GBT)")
print(f"    feature_importances.json  (impurity + permutation, both methods)")
print(f"    metrics.json              (full nested CV + test set results)")
print(f"    js_weights.json           (dashboard-ready weights + metadata)")
print(f"    dataset_sample.csv        (first 200 rows of training data)")
print(f"{'='*65}\n")
