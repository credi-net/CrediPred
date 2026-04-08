"""Conformal prediction scoring functions for classification.

Implements APS (Adaptive Prediction Sets) and RAPS (Regularized APS)
following the original CF-GNN approach (snap-stanford/conformalized-gnn).
"""

import numpy as np


def aps(
    cal_smx: np.ndarray,
    val_smx: np.ndarray,
    cal_labels: np.ndarray,
    val_labels: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, float, float]:
    """Adaptive Prediction Sets.

    Args:
        cal_smx: calibration softmax probabilities [n_cal, num_classes].
        val_smx: validation/test softmax probabilities [n_val, num_classes].
        cal_labels: calibration ground-truth labels [n_cal].
        val_labels: validation/test ground-truth labels [n_val].
        alpha: miscoverage rate (e.g. 0.1 for 90% coverage target).

    Returns:
        (prediction_sets, coverage, efficiency, qhat)
    """
    n = cal_smx.shape[0]
    # Sort classes by descending probability, accumulate
    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
        np.arange(n), cal_labels
    ]
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )
    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(
        val_srt <= qhat, val_pi.argsort(axis=1), axis=1
    )
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, float(cov), float(eff), float(qhat)


def raps(
    cal_smx: np.ndarray,
    val_smx: np.ndarray,
    cal_labels: np.ndarray,
    val_labels: np.ndarray,
    alpha: float,
    lam_reg: float = 0.01,
    k_reg: int = 1,
) -> tuple[np.ndarray, float, float, float]:
    """Regularized Adaptive Prediction Sets.

    Args:
        cal_smx: calibration softmax probabilities [n_cal, num_classes].
        val_smx: validation/test softmax probabilities [n_val, num_classes].
        cal_labels: calibration ground-truth labels [n_cal].
        val_labels: validation/test ground-truth labels [n_val].
        alpha: miscoverage rate.
        lam_reg: regularization strength.
        k_reg: number of top classes exempt from regularization.

    Returns:
        (prediction_sets, coverage, efficiency, qhat)
    """
    n = cal_smx.shape[0]
    n_val = val_smx.shape[0]
    num_classes = cal_smx.shape[1]
    k_reg = min(k_reg, num_classes)
    reg_vec = np.array(k_reg * [0.0] + (num_classes - k_reg) * [lam_reg])[None, :]

    cal_pi = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    cal_scores = (
        cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]
        - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
    )
    qhat = np.quantile(
        cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, method="higher"
    )

    val_pi = val_smx.argsort(1)[:, ::-1]
    val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
    val_srt_reg = val_srt + reg_vec
    indicators = (
        val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg
    ) <= qhat
    prediction_sets = np.take_along_axis(
        indicators, val_pi.argsort(axis=1), axis=1
    )
    cov = prediction_sets[np.arange(prediction_sets.shape[0]), val_labels].mean()
    eff = np.sum(prediction_sets) / len(prediction_sets)
    return prediction_sets, float(cov), float(eff), float(qhat)


def run_conformal_eval(
    softmax_probs: np.ndarray,
    labels: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    score: str = "aps",
    n_trials: int = 100,
    calib_fraction: float = 0.5,
) -> tuple[float, float, float]:
    """Run conformal evaluation with random calibration/test splits.

    Follows the original CF-GNN protocol: split the given indices into
    calibration and evaluation halves, repeat n_trials times, average.

    Args:
        softmax_probs: softmax probabilities for ALL nodes [N, num_classes].
        labels: ground-truth labels for ALL nodes [N].
        cal_idx: indices of nodes available for calibration+evaluation.
        test_idx: indices of nodes available for calibration+evaluation.
        alpha: miscoverage rate.
        score: 'aps' or 'raps'.
        n_trials: number of random splits.
        calib_fraction: fraction used for calibration in each trial.

    Returns:
        (mean_coverage, mean_efficiency, mean_qhat)
    """
    # Combine cal_idx and test_idx for random splitting
    all_idx = np.concatenate([cal_idx, test_idx])
    smx = softmax_probs[all_idx]
    lbls = labels[all_idx]
    n_cal = int(len(all_idx) * calib_fraction)

    cov_all, eff_all, qhat_all = [], [], []
    score_fn = aps if score == "aps" else raps

    for k in range(n_trials):
        idx = np.array([1] * n_cal + [0] * (len(all_idx) - n_cal)) > 0
        np.random.seed(k)
        np.random.shuffle(idx)
        cal_smx, val_smx = smx[idx], smx[~idx]
        cal_labels, val_labels = lbls[idx], lbls[~idx]

        _, cov, eff, qhat = score_fn(cal_smx, val_smx, cal_labels, val_labels, alpha)
        cov_all.append(cov)
        eff_all.append(eff)
        qhat_all.append(qhat)

    return float(np.mean(cov_all)), float(np.mean(eff_all)), float(np.mean(qhat_all))
