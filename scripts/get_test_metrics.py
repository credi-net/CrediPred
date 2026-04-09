import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, cast

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import NeighborLoader
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm import tqdm

from credipred.dataset.dataset import DATASETS, WebGraphDataset
from credipred.encoders.encoder import Encoder
from credipred.encoders.encoders import ENCODERS
from credipred.gnn.model import Model
from credipred.utils.args import MetaArguments, ModelArguments, parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir, get_scratch
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Get test metrics.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)
parser.add_argument(
    '--category',
    choices=['malware', 'general', 'phishing', 'misinfo', 'none'],
    default='none',
    help='Whether to use and what subcategory to filer.',
)


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
        mask = (confidences > bin_boundaries[i]) & (
            confidences <= bin_boundaries[i + 1]
        )
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


def get_binary_metrics(
    model_arguments: ModelArguments,
    dataset: WebGraphDataset,
    weight_directory: Path,
    annotation_dir: Path,
    category_filter_test: str = 'none',
) -> None:
    get_root_dir()
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    test_idx = dataset.get_idx_split()['test']
    domain_to_idx_mapping = dataset.get_mapping()
    idx_to_domain_mapping = {v: k for k, v in domain_to_idx_mapping.items()}
    logging.info('Mapping returned.')
    model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
        binary=True,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info('Model Loaded.')
    model.eval()

    if category_filter_test != 'none':
        domain_rel_annotations_dict = pickle.load(
            open(annotation_dir / 'domain_rel_annotations_dict.pkl', 'rb')
        )
        domain_rel_annotations_dict = {
            k: v[0] for k, v in domain_rel_annotations_dict.items()
        }
        category_set = set(
            [
                k
                for k, v in domain_rel_annotations_dict.items()
                if v == category_filter_test
            ]
        )

        mask = torch.tensor(
            [
                idx.item() in idx_to_domain_mapping
                and idx_to_domain_mapping[idx.item()] in category_set
                for idx in test_idx
            ],
            dtype=torch.bool,
        )

        all_domains = len(test_idx)
        test_idx = test_idx[mask]
        category_domains = len(test_idx)
        logging.info(
            f'Test indices filter within: {category_filter_test}. All domains in test set: {all_domains}, {category_filter_test} domains: {category_domains}'
        )

    test_targets = dataset[0].y[test_idx]
    count_ones = 0
    count_zeros = 0
    for pred in test_targets:
        if pred == 1:
            count_ones += 1
        else:
            count_zeros += 1

    logging.info(f'Target distribution. Ones: {count_ones}, zeros: {count_zeros}')

    loader = NeighborLoader(
        data,
        input_nodes=test_idx,
        num_neighbors=[30, 30, 30],
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )
    num_nodes = data.num_nodes
    all_preds = torch.zeros(num_nodes, 2)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Inference'):
            batch = batch.to(device)
            preds = model.forward(batch.x, batch.edge_index)
            seed_nodes = batch.n_id[: batch.batch_size]
            all_preds[seed_nodes] = preds[: batch.batch_size].cpu()

    test_logits = all_preds[test_idx]

    predicted_labels = torch.argmax(test_logits, dim=1)
    bcm = BinaryConfusionMatrix()
    conf_matrix = bcm(predicted_labels, test_targets)

    logging.info(f'Confusion Matrix: \n{conf_matrix}')
    tn, fp, fn, tp = conf_matrix.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    f1_score = 2 * ((precision * recall) / (precision + recall))

    logging.info(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n')
    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'Recall: {recall}')
    logging.info(f'Precision: {precision}')
    logging.info(f'f1_score: {f1_score}')

    test_targets_np = test_targets.numpy()
    test_probs = torch.exp(test_logits[:, 1]).numpy()
    predicted_labels_np = predicted_labels.numpy()

    roc_auc = roc_auc_score(test_targets_np, predicted_labels_np)
    pr_auc = average_precision_score(test_targets_np, predicted_labels_np)

    logging.info(f'ROC-AUC: {roc_auc:.4f}')
    logging.info(f'PR-AUC (Average Precision): {pr_auc:.4f}')

    # --- Uncertainty Quantification ---
    test_probs = torch.exp(test_logits)
    base_ece = compute_ece(test_probs.numpy(), test_targets_np)
    base_nll = compute_nll(test_probs.numpy(), test_targets_np)
    logging.info('  ECE:      %.4f  (lower = better calibrated)', base_ece)
    logging.info('  NLL:      %.4f  (lower = better probability estimates)', base_nll)


def build_experiment_dataset(
    meta_args: MetaArguments, root: Path, encoding_dict: Dict[str, Encoder]
) -> WebGraphDataset:
    if meta_args.is_regression:
        dataset_type = 'Regression'
    else:
        dataset_type = 'BinaryGlobal'

    cfg = vars(meta_args).copy()

    return DATASETS.build(
        cfg={**cfg, 'type': dataset_type},
        root=root,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
    )


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(cast(str, meta_args.log_file_path) + 'GET_EMBEDDINGS.log')
    seed_everything(meta_args.global_seed)

    encoding_dict = {
        idx: ENCODERS.build(
            {'type': val, 'dimension': meta_args.initalization_dimension}
        )
        for idx, val in meta_args.encoder_dict.items()
    }
    logging.info(f'Encoding Dictionary: {encoding_dict}')

    dataset = build_experiment_dataset(meta_args, root, encoding_dict)
    logging.info('In-Memory Dataset loaded.')
    logging.info(f'Dataset {type(dataset).__name__} loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    annotation_dir = scratch / 'data' / 'splits' / 'hussien' / 'sub_category'

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        get_binary_metrics(
            experiment_arg.model_args,
            dataset,
            weight_directory,
            annotation_dir,
            args.category,
        )


if __name__ == '__main__':
    main()
