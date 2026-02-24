import argparse
import logging
from pathlib import Path
from typing import Dict, cast

import torch
from torch_geometric.loader import NeighborLoader
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm import tqdm

from credipred.dataset.temporal_dataset import (
    TemporalBinaryDatasetAllGlobalSplits,
)
from credipred.encoders.categorical_encoder import CategoricalEncoder
from credipred.encoders.encoder import Encoder
from credipred.encoders.norm_encoding import NormEncoder
from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.encoders.zero_encoder import ZeroEncoder
from credipred.gnn.model import Model
from credipred.utils.args import ModelArguments, parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir, get_scratch
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Get Confusion Matrix Binary Experiment.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def run_get_test_predictions(
    model_arguments: ModelArguments,
    dataset: TemporalBinaryDatasetAllGlobalSplits,
    weight_directory: Path,
    target: str,
) -> None:
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    test_idx = dataset.get_idx_split()['test']
    logging.info(f'Length of testing indices: {len(test_idx)}')
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

    test_targets = dataset[0].y[test_idx]
    logging.info(f'Target values: {test_targets}')
    indices = torch.tensor(test_idx, dtype=torch.long)

    loader = NeighborLoader(
        data,
        input_nodes=indices,
        num_neighbors=[30, 30, 30],
        batch_size=1024,
        shuffle=False,
    )
    logging.info(f'Test indices loader  created for {len(indices)} nodes.')

    num_nodes = data.num_nodes
    all_preds = torch.zeros(num_nodes, 2)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'batch'):
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index)
            seed_nodes = batch.n_id[: batch.batch_size]
            all_preds[seed_nodes] = preds[: batch.batch_size].cpu()

    test_logits = all_preds[indices]

    predicted_labels = torch.argmax(test_logits, dim=1)

    bcm = BinaryConfusionMatrix()
    conf_matrix = bcm(predicted_labels, test_targets)

    logging.info(f'Confusion Matrix: \n{conf_matrix}')

    tn, fp, fn, tp = conf_matrix.ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 * ((precision * recall) / (precision + recall))

    logging.info(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n')
    logging.info(f'f1_score: {f1_score}')


def main() -> None:
    root = get_root_dir()
    get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(str(meta_args.log_file_path) + ':_GET_F1.log')
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
        'ZERO': ZeroEncoder(64),
        'NORM': NormEncoder(),
        'CAT': CategoricalEncoder(),
        'PRE': TextEmbeddingEncoder(64),
    }

    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoder_class = encoder_classes[value]
        encoding_dict[index] = encoder_class

    dataset = TemporalBinaryDatasetAllGlobalSplits(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
        processed_dir=cast(str, meta_args.processed_location),
        split_dir=cast(str, meta_args.split_folder),
        embedding_lookup=meta_args.embedding_lookup,
        embedding_location=cast(str, meta_args.embedding_location),
    )
    logging.info('In-Memory Dataset loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_get_test_predictions(
            experiment_arg.model_args,
            dataset,
            weight_directory,
            target=meta_args.target_col,
        )


if __name__ == '__main__':
    main()
