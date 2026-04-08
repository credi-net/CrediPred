"""Evaluate classification UQ metrics from cached predictions — no training needed.

Usage:
    uv run python credipred/experiments/gnn_experiments/evaluate_uq_metrics.py \
        --config-file configs/gnn/classification/gat_domainrel_classification_uq_dec.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch

from credipred.utils.args import parse_args
from credipred.utils.path import get_root_dir

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(smx: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual correctness.
    Bins samples by confidence (max softmax prob), computes
    |accuracy - confidence| per bin, weighted by bin size.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better. 0 = perfectly calibrated.
    """
    confidences = smx.max(axis=1)
    predictions = smx.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)
    return ece


def compute_nll(smx: np.ndarray, labels: np.ndarray) -> float:
    """Negative Log-Likelihood (NLL).

    NLL = -mean(log(p(y_true)))

    Measures quality of predicted probability for the true class.
    Lower is better. Equivalent to cross-entropy on test set.
    """
    probs_clipped = np.clip(smx, 1e-7, 1.0)
    return float(-np.log(probs_clipped[np.arange(len(labels)), labels]).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate classification UQ metrics')
    parser.add_argument('--config-file', required=True)
    args = parser.parse_args()

    root = get_root_dir()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)

    for exp_name, exp_arg in experiment_args.exp_args.items():
        weight_dir = root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}' / exp_arg.model_args.model
        alpha = exp_arg.model_args.conformal_alpha

    logging.info('Weight dir: %s', weight_dir)
    logging.info('Alpha: %.4f', alpha)

    # ---- Load cached base softmax ----
    cache_path = weight_dir / 'base_softmax.pt'
    if not cache_path.exists():
        logging.error('base_softmax.pt not found at %s', cache_path)
        sys.exit(1)
    base_softmax = torch.load(cache_path, map_location='cpu', weights_only=True)
    logging.info('Loaded base softmax: shape=%s', base_softmax.shape)

    # ---- Load corrected softmax if available ----
    corr_path = weight_dir / 'corrected_softmax.pt'
    has_corrected = corr_path.exists()
    if has_corrected:
        corr_softmax = torch.load(corr_path, map_location='cpu', weights_only=True)
        logging.info('Loaded corrected softmax')

    # ---- Load dataset for labels and splits ----
    from credipred.dataset.temporal_dataset import TemporalBinaryDatasetAllGlobalSplits
    from credipred.conformal_classification.conformal import run_conformal_eval
    from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder

    encoder_dict = {}
    for index, value in meta_args.encoder_dict.items():
        if value == 'PRE':
            encoder_dict[index] = TextEmbeddingEncoder(64)

    dataset = TemporalBinaryDatasetAllGlobalSplits(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        split_dir=cast(str, meta_args.split_folder),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        force_undirected=meta_args.force_undirected,
        switch_source=meta_args.switch_source,
        encoding=encoder_dict,
        seed=meta_args.global_seed,
        processed_dir=cast(str, meta_args.processed_location),
        embedding_location=cast(str, meta_args.embedding_location),
        embedding_lookup=cast(str, meta_args.embedding_lookup),
    )
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    test_idx = split_idx['test'].numpy()
    val_idx = split_idx['valid'].numpy()
    labels_np = data.y.cpu().numpy()
    base_smx_np = base_softmax.numpy()

    logging.info('Test: %d, Val: %d', len(test_idx), len(val_idx))

    # ---- Base model ----
    logging.info('========== BASE MODEL ==========')
    base_acc = (base_smx_np[test_idx].argmax(1) == labels_np[test_idx]).mean()
    base_ece = compute_ece(base_smx_np[test_idx], labels_np[test_idx])
    base_nll = compute_nll(base_smx_np[test_idx], labels_np[test_idx])
    logging.info('  Accuracy: %.4f', base_acc)
    logging.info('  ECE:      %.4f  (lower = better calibrated)', base_ece)
    logging.info('  NLL:      %.4f  (lower = better probability estimates)', base_nll)

    for score in ['aps', 'raps']:
        cov, eff, qhat = run_conformal_eval(
            base_smx_np, labels_np, val_idx, test_idx, alpha, score=score,
        )
        logging.info('  %s: coverage=%.4f efficiency=%.4f qhat=%.4f', score.upper(), cov, eff, qhat)

    # ---- Corrected model (if available) ----
    if has_corrected:
        corr_smx_np = corr_softmax.numpy()
        logging.info('========== CORRECTED MODEL (ConfGNN) ==========')
        corr_acc = (corr_smx_np[test_idx].argmax(1) == labels_np[test_idx]).mean()
        corr_ece = compute_ece(corr_smx_np[test_idx], labels_np[test_idx])
        corr_nll = compute_nll(corr_smx_np[test_idx], labels_np[test_idx])
        logging.info('  Accuracy: %.4f', corr_acc)
        logging.info('  ECE:      %.4f', corr_ece)
        logging.info('  NLL:      %.4f', corr_nll)

        for score in ['aps', 'raps']:
            cov, eff, qhat = run_conformal_eval(
                corr_smx_np, labels_np, val_idx, test_idx, alpha, score=score,
            )
            logging.info('  %s: coverage=%.4f efficiency=%.4f qhat=%.4f', score.upper(), cov, eff, qhat)

        logging.info('========== COMPARISON ==========')
        logging.info('  Accuracy: %.4f → %.4f (%+.4f)', base_acc, corr_acc, corr_acc - base_acc)
        logging.info('  ECE:      %.4f → %.4f (%+.4f)', base_ece, corr_ece, corr_ece - base_ece)
        logging.info('  NLL:      %.4f → %.4f (%+.4f)', base_nll, corr_nll, corr_nll - base_nll)
    else:
        logging.info('No corrected_softmax.pt found — showing base model metrics only.')


if __name__ == '__main__':
    main()
