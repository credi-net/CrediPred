import argparse
import logging
import pathlib
import pdb
from typing import Dict, cast

from credipred.dataset.dataset import WebGraphDataset
from credipred.encoders.encoder import Encoder
from credipred.experiments.end_to_end.end_to_end_experiment_binary import (
    run_end_to_end_binary_classification,
)
from credipred.utils.args import MetaArguments, parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir
from credipred.utils.readers import get_embeddings_lookup
from credipred.utils.registry import DATASETS, ENCODERS
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='End-To-End MLP(TEXT + GNN) Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
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


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    pdb.set_trace()
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoding_dict = {
        idx: ENCODERS.build({'type': val, 'dim': 64})
        for idx, val in meta_args.encoder_dict.items()
    }
    logging.info(f'Encoding Dictionary: {encoding_dict}')

    logging.info('In-Memory Dataset loaded.')

    dataset = build_experiment_dataset(meta_args, root, encoding_dict)
    logging.info(f'Dataset {type(dataset).__name__} loaded.')

    weights_path = (
        root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}'
    )

    if args.embedding_location:
        embeddings_location = pathlib.Path(args.embedding_location)
    else:
        embeddings_location = pathlib.Path()

    logging.info(f'Embedding location: {embeddings_location}')
    embedding_lookup = args.embedding_lookup

    embeddings_lookup_table = get_embeddings_lookup(
        str(embeddings_location / embedding_lookup)
    )
    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        run_end_to_end_binary_classification(
            experiment_arg.data_args,
            experiment_arg.model_args,
            weights_path,
            dataset,
            embeddings_location,
            embeddings_lookup_table,
        )

    logging.info('***Experiments Complete.***')


if __name__ == '__main__':
    main()
