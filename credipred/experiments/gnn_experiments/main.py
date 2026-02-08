import argparse
import logging
from typing import Dict, List, cast

from credipred.dataset.squashed_dataset import (
    SquashedBinaryDatasetAllGlobalSplits,
    SquashedDatasetGlobalSplit,
)
from credipred.dataset.temporal_dataset import (
    TemporalBinaryDatasetAllGlobalSplits,
    TemporalDatasetGlobalSplit,
)
from credipred.encoders.categorical_encoder import CategoricalEncoder
from credipred.encoders.encoder import Encoder
from credipred.encoders.multi_snapshot_text_encoder import (
    MultiSnapTextEmbeddingEncoder,
)
from credipred.encoders.norm_encoding import NormEncoder
from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.encoders.zero_encoder import ZeroEncoder
from credipred.experiments.gnn_experiments.gnn_experiment import (
    run_gnn_baseline,
)
from credipred.experiments.gnn_experiments.gnn_experiment_binary_labels import (
    run_binary_class_gnn_baseline,
)
from credipred.utils.args import parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir
from credipred.utils.plot import (
    load_all_loss_tuples,
    plot_metric_across_models,
    plot_metric_per_encoder,
    plot_model_per_encoder,
)
from credipred.utils.seed import seed_everything

parser = argparse.ArgumentParser(
    description='GNN Experiments.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--binary-classification',
    action='store_true',
    help='Whether to use binary classification, otherwise regression is used in training.',
)
parser.add_argument(
    '--squash',
    action='store_true',
    help='Whether to squash all snapshots into a large static graph.',
)
parser.add_argument(
    '--config-file',
    type=str,
    default='configs/gnn/base.yaml',
    help='Path to yaml configuration file to use',
)


def main() -> None:
    root = get_root_dir()
    args = parser.parse_args()
    config_file_path = root / args.config_file
    meta_args, experiment_args = parse_args(config_file_path)
    setup_logging(meta_args.log_file_path)
    seed_everything(meta_args.global_seed)

    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),  # TODO: Set this a paramater
        'ZERO': ZeroEncoder(64),
        'NORM': NormEncoder(),
        'CAT': CategoricalEncoder(),
        'PRE': TextEmbeddingEncoder(64),
        'MULTI': MultiSnapTextEmbeddingEncoder(64),
    }

    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoder_class = encoder_classes[value]
        encoding_dict[index] = encoder_class

    logging.info(f'Encoding Dictionary: {encoding_dict}')

    logging.info(f'force_undirected: {meta_args.force_undirected}')

    if not args.squash:
        if args.binary_classification:
            logging.info('Task in use: Classification')
            logging.info(f'Using global splits: {cast(str, meta_args.split_folder)}')
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
            )  # Map to .to_cpu()
        else:
            logging.info('Task in use: Regression')
            dataset = TemporalDatasetGlobalSplit(
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
            )  # Map to .to_cpu()
    else:
        logging.info('Using squashed dataset.')
        if args.binary_classification:
            logging.info('Task in use: Classification')
            logging.info(f'Using global splits: {cast(str, meta_args.split_folder)}')
            dataset = SquashedBinaryDatasetAllGlobalSplits(
                root=f'{root}/data/',
                node_files=cast(List[str], meta_args.node_file),
                edge_files=cast(List[str], meta_args.edge_file),
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
                embedding_location=cast(List[str], meta_args.embedding_location),
                embedding_lookup=cast(List[str], meta_args.embedding_lookup),
            )  # Map to .to_cpu()
        else:
            logging.info('Task in use: Regression')
            logging.info(f'Using global splits: {cast(str, meta_args.split_folder)}')
            dataset = SquashedDatasetGlobalSplit(
                root=f'{root}/data/',
                node_files=cast(List[str], meta_args.node_file),
                edge_files=cast(List[str], meta_args.edge_file),
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
                embedding_location=cast(List[str], meta_args.embedding_location),
                embedding_lookup=cast(List[str], meta_args.embedding_lookup),
            )  # Map to .to_cpu()

    logging.info('In-Memory Dataset loaded.')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        if not args.binary_classification:
            run_gnn_baseline(
                experiment_arg.data_args,
                experiment_arg.model_args,
                root
                / cast(str, meta_args.weights_directory)
                / f'{meta_args.target_col}',
                dataset,
            )
        else:
            run_binary_class_gnn_baseline(
                experiment_arg.data_args,
                experiment_arg.model_args,
                root
                / cast(str, meta_args.weights_directory)
                / f'{meta_args.target_col}',
                dataset,
            )

    results = load_all_loss_tuples()
    logging.info('Constructing Plots, across models')
    plot_metric_across_models(results)
    logging.info('Constructing Plots, metric per-encoder')
    plot_metric_per_encoder(results)
    logging.info('Constructing Plots, model per-encoder')
    plot_model_per_encoder(results)


if __name__ == '__main__':
    main()
