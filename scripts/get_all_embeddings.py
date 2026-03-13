import argparse
import logging
from pathlib import Path
from typing import Dict, cast

import pandas as pd
import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import (
    TemporalBinaryDatasetAllGlobalSplits,
)
from credipred.encoders.encoder import Encoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.gnn.model import Model
from credipred.utils.args import ModelArguments, parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir, get_scratch
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Return all embeddings per snapshot.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def get_embeddings(
    model_arguments: ModelArguments,
    dataset: TemporalBinaryDatasetAllGlobalSplits,
    weight_directory: Path,
    scratch: Path,
) -> None:
    get_root_dir()
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    domain_to_idx_mapping = dataset.get_mapping()
    idx_to_domain_mapping = {
        v: k for k, v in domain_to_idx_mapping.items()
    }  # One time, to speed up lookup
    tensor_idx = torch.tensor(list(domain_to_idx_mapping.values()))
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

    num_nodes = data.num_nodes
    all_preds_embeddings = torch.zeros(num_nodes, 256)

    loader = NeighborLoader(
        data,
        input_nodes=tensor_idx,
        num_neighbors=[30, 30, 30],
        batch_size=4096,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )

    shard_size = 3_000_000
    current_shard_dict = {}
    master_index_dict = {}
    shard_count = 0

    save_dir = scratch / 'shards'
    logging.info(f'save directory: {save_dir}')
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Inference'):
            batch = batch.to(device)
            preds = model.get_embeddings(batch.x, batch.edge_index)
            seed_nodes = batch.n_id[: batch.batch_size]
            all_preds_embeddings[seed_nodes] = preds[: batch.batch_size].cpu()

            for i, node_idx in enumerate(seed_nodes):
                name = idx_to_domain_mapping[node_idx.item()]
                embedding = preds[i].cpu().tolist()
                current_shard_dict[name] = embedding

            if len(current_shard_dict) >= shard_size:
                shard_filename = f'shard_{shard_count}.parquet'
                shard_path = save_dir / shard_filename
                for domain in current_shard_dict.keys():
                    master_index_dict[domain] = shard_filename

                df = pd.DataFrame(
                    [
                        {'domain': k, 'embedding': v}
                        for k, v in current_shard_dict.items()
                    ]
                )
                df.to_parquet(shard_path, index=False)

                logging.info(
                    f'Saved shard {shard_count} with {len(current_shard_dict)}'
                )

                current_shard_dict = {}
                shard_count += 1

        if current_shard_dict:
            shard_filename = f'shard_{shard_count}.parquet'
            shard_path = save_dir / shard_filename
            for domain in current_shard_dict.keys():
                master_index_dict[domain] = shard_filename
            df = pd.DataFrame(
                [{'domain': k, 'embedding': v} for k, v in current_shard_dict.items()]
            )
            df.to_parquet(shard_path, index=False)

    index_path = save_dir / 'domain_index.parquet'
    pd.DataFrame(
        [{'domain': k, 'index_file': v} for k, v in master_index_dict.items()]
    ).to_parquet(index_path, index=False)

    logging.info(f'Saved domain embeddings and index to {save_dir}')


def main() -> None:
    root = get_root_dir()
    scratch = get_scratch()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
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
        split_dir=cast(str, meta_args.split_folder),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        force_undirected=meta_args.force_undirected,
        switch_source=meta_args.switch_source,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
        processed_dir=cast(str, meta_args.processed_location),
        embedding_location=cast(str, meta_args.embedding_location),
        embedding_lookup=cast(str, meta_args.embedding_lookup),
    )
    logging.info('In-Memory Dataset loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        get_embeddings(
            experiment_arg.model_args,
            dataset,
            weight_directory,
            scratch / cast(str, meta_args.database_folder),
        )


if __name__ == '__main__':
    main()
