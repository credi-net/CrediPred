"""Evaluate Max/Min Absolute Error from saved model weights.

Usage:
    uv run python credipred/experiments/gnn_experiments/get_max_ae.py \
        --config-file configs/gnn/gat_dec.yaml
"""

import argparse
import logging
import pathlib
from typing import Any, Dict, cast

import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.dataset import DATASETS, WebGraphDataset
from credipred.encoders.encoder import Encoder
from credipred.encoders.encoders import ENCODERS
from credipred.gnn.model import Model
from credipred.utils.args import MetaArguments, ModelArguments, parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir
from credipred.utils.seed import seed_everything

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s'
)

parser = argparse.ArgumentParser(
    description='Evaluate Max/Min Absolute Error from saved weights.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    required=True,
    help='Path to yaml configuration file to use',
)


def build_experiment_dataset(
    meta_args: MetaArguments, root: pathlib.Path, encoding_dict: Dict[str, Encoder]
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


def _build_kwargs(model_name: str) -> Dict[str, Any]:
    if model_name == 'GPS':
        return {
            'gps_head': 4,
            'gps_attn_type': 'performer',
            'gps_local_mpnn': 'gin',
        }
    return {}


def evaluate_max_ae(
    model_arguments: ModelArguments,
    dataset: WebGraphDataset,
    weight_directory: pathlib.Path,
) -> None:
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device: {device}')

    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    logging.info(f'Loading weights from: {weight_path}')

    kwargs = _build_kwargs(model_arguments.model)
    model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
        binary=False,
        **kwargs,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    logging.info('Model loaded.')

    test_idx = split_idx['test']
    test_targets = data.y[test_idx]

    loader = NeighborLoader(
        data,
        input_nodes=test_idx,
        num_neighbors=model_arguments.num_neighbors,
        batch_size=256,
        shuffle=False,
        num_workers=4,
    )

    num_nodes = data.num_nodes
    all_preds = torch.zeros(num_nodes, 1)

    batch_kwarg = {'batch': None} if model_arguments.model == 'GPS' else {}

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, **batch_kwarg)
            seed_nodes = batch.n_id[: batch.batch_size]
            all_preds[seed_nodes] = preds[: batch.batch_size].cpu()

    test_preds = all_preds[test_idx].squeeze()
    # Filter out invalid targets (e.g. -1 for unlabeled nodes)
    valid_mask = test_targets >= 0
    test_preds = test_preds[valid_mask]
    test_targets = test_targets[valid_mask]
    logging.info(f'Valid test nodes: {valid_mask.sum().item()} / {len(valid_mask)}')
    abs_errors = (test_preds - test_targets).abs()

    logging.info(f'--- {model_arguments.model} ---')
    logging.info(f'Min Absolute Error: {abs_errors.min().item():.6f}')
    logging.info(f'Max Absolute Error: {abs_errors.max().item():.6f}')
    logging.info(f'Mean Absolute Error: {abs_errors.mean().item():.6f}')
    logging.info(f'Median Absolute Error: {abs_errors.median().item():.6f}')
    logging.info(f'Std Absolute Error: {abs_errors.std().item():.6f}')


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoding_dict = {
        idx: ENCODERS.build(
            {'type': val, 'dimension': meta_args.initalization_dimension}
        )
        for idx, val in meta_args.encoder_dict.items()
    }

    dataset = build_experiment_dataset(meta_args, root, encoding_dict)
    logging.info(f'Dataset {type(dataset).__name__} loaded.')

    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Evaluating**: {experiment}')
        evaluate_max_ae(
            experiment_arg.model_args,
            dataset,
            weight_directory,
        )


if __name__ == '__main__':
    main()
