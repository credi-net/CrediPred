import argparse
import logging
from pathlib import Path
from typing import Dict, cast

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
from credipred.utils.path import get_root_dir
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


def get_binary_metrics(
    model_arguments: ModelArguments,
    dataset: WebGraphDataset,
    weight_directory: Path,
) -> None:
    get_root_dir()
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    test_idx = dataset.get_idx_split()['test']
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
    test_indices = torch.tensor(test_idx, dtype=torch.long)

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
        input_nodes=test_indices,
        num_neighbors=[30, 30, 30],
        batch_size=4096,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
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

    test_logits = all_preds[test_indices]

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

    roc_auc = roc_auc_score(test_targets_np, test_probs)
    pr_auc = average_precision_score(test_targets_np, test_probs)

    logging.info(f'ROC-AUC: {roc_auc:.4f}')
    logging.info(f'PR-AUC (Average Precision): {pr_auc:.4f}')


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

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        get_binary_metrics(
            experiment_arg.model_args,
            dataset,
            weight_directory,
        )


if __name__ == '__main__':
    main()
