import argparse
import logging
from pathlib import Path
from typing import Dict, List, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalDatasetGlobalSplit
from credipred.encoders.categorical_encoder import CategoricalEncoder
from credipred.encoders.encoder import Encoder
from credipred.encoders.norm_encoding import NormEncoder
from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.encoders.zero_encoder import ZeroEncoder
from credipred.gnn.model import Model
from credipred.utils.args import ModelArguments, parse_args
from credipred.utils.domain_handler import reverse_domain
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='Get final dqr node embeddings.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def run_forward_get_embeddings(
    model_arguments: ModelArguments,
    dataset: TemporalDatasetGlobalSplit,
    weight_directory: Path,
) -> None:
    root = get_root_dir()
    dqr_dict: Dict[str, str] = {
        'DQR Nodes': 'data/dqr/domain_pc1.csv',
    }
    data = dataset[0]
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    logging.info(f'Device found: {device}')
    weight_path = weight_directory / f'{model_arguments.model}' / 'best_model.pt'
    mapping = dataset.get_mapping()
    logging.info('Mapping returned.')
    model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
        binary=False,
    ).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    logging.info('Model Loaded.')
    model.eval()
    for dataset_name, path in dqr_dict.items():
        logging.info(f'Getting final node embeddings of: {dataset_name}')
        df = pd.read_csv(root / path)
        indices = [
            mapping.get(reverse_domain(domain.strip())) for domain in df['domain']
        ]
        indices = [i for i in indices if i is not None]
        indices = torch.tensor(indices, dtype=torch.long)

        loader = NeighborLoader(
            data,
            input_nodes=indices,
            num_neighbors=[30, 30, 30],
            batch_size=1024,
            shuffle=False,
        )
        logging.info(f'{dataset_name}: loader  created for {len(indices)} nodes.')

        num_nodes = data.num_nodes
        all_preds_embeddings = torch.zeros(num_nodes, 256)

        with torch.no_grad():
            for batch in tqdm(loader, desc=f'{dataset_name} batch'):
                batch = batch.to(device)
                embeddings = model.get_embeddings(batch.x, batch.edge_index)
                seed_nodes = batch.n_id[: batch.batch_size]
                all_preds_embeddings[seed_nodes] = embeddings[: batch.batch_size].cpu()

        domain_embeddings = {}

        for domain in df['domain']:
            dom = domain.strip()
            idx = mapping.get(reverse_domain(dom))
            if idx is not None:
                domain_embeddings[dom] = all_preds_embeddings[idx].tolist()

        parquet_rows: Dict[str, List] = {'domain': [], 'embeddings': []}

        for domain, emb in domain_embeddings.items():
            parquet_rows['domain'].append(domain)
            parquet_rows['embeddings'].append(emb)

        write_domain_emb_parquet(
            rows=parquet_rows,
            directory_path=weight_directory,
            file_name='dqr_domain_gat_from_text_embeddings_updated.parquet',
        )


def write_domain_emb_parquet(rows: Dict, directory_path: Path, file_name: str) -> None:
    schema = pa.schema(
        [
            ('domain', pa.string()),
            (
                'embeddings',
                pa.list_(pa.float32()),
            ),
        ]
    )
    table = pa.Table.from_pydict(rows, schema=schema)
    table = table.sort_by('domain')
    pq.write_table(
        table,
        directory_path / file_name,
        row_group_size=100,
        use_dictionary=['domain'],
    )
    logging.info(f'Saved domain embedding to {directory_path / file_name}')


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging('DQR_EMBEDDINGS:_' + str(meta_args.log_file_path))
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

    dataset = TemporalDatasetGlobalSplit(
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
        embedding_location=cast(str, meta_args.embedding_location),
        embedding_lookup=meta_args.embedding_lookup,
    )
    logging.info('In-Memory Dataset loaded.')
    weight_directory = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_forward_get_embeddings(
            experiment_arg.model_args,
            dataset,
            weight_directory,
        )


if __name__ == '__main__':
    main()
