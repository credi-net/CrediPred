import argparse
import logging
import os
from typing import Dict, cast

import wandb

from credipred.dataset.temporal_dataset import TemporalDataset
from credipred.encoders.categorical_encoder import CategoricalEncoder
from credipred.encoders.encoder import Encoder
from credipred.encoders.norm_encoding import NormEncoder
from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.encoders.zero_encoder import ZeroEncoder
from credipred.experiments.gnn_experiments.gnn_experiment import (
    run_gnn_baseline,
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

    # Initialize wandb with comprehensive config (offline mode - no API key required)
    wandb_run = wandb.init(
        project="CrediPred-GNN",
        name=f"{meta_args.target_col}-{os.path.basename(args.config_file).replace('.yaml', '')}",
        config={
            # Meta arguments
            "config_file": args.config_file,
            "target_col": meta_args.target_col,
            "node_file": meta_args.node_file,
            "edge_file": meta_args.edge_file,
            "global_seed": meta_args.global_seed,
            "force_undirected": meta_args.force_undirected,
            "encoder_dict": meta_args.encoder_dict,
        },
        tags=[meta_args.target_col, os.path.basename(args.config_file).replace('.yaml', '')],
        save_code=True,
        mode="offline",
    )
    logging.info(f"WandB run initialized: {wandb_run.url}")

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

    logging.info(f'Encoding Dictionary: {encoding_dict}')

    logging.info(f'force_undirected: {meta_args.force_undirected}')

    dataset = TemporalDataset(
        root=f'{root}/data/',
        node_file=cast(str, meta_args.node_file),
        edge_file=cast(str, meta_args.edge_file),
        target_file=cast(str, meta_args.target_file),
        target_col=meta_args.target_col,
        edge_src_col=meta_args.edge_src_col,
        edge_dst_col=meta_args.edge_dst_col,
        index_col=meta_args.index_col,
        force_undirected=meta_args.force_undirected,
        switch_source=meta_args.switch_source,
        encoding=encoding_dict,
        seed=meta_args.global_seed,
        processed_dir=cast(str, meta_args.processed_location),
        embedding_index_file=meta_args.embedding_index_file,
        embedding_folder=meta_args.embedding_folder,
    )  # Map to .to_cpu()
    logging.info('In-Memory Dataset loaded.')

    for experiment, experiment_arg in experiment_args.exp_args.items():
        logging.info(f'\n**Running**: {experiment}')
        # Update wandb config with experiment-specific parameters
        wandb.config.update({
            f"experiment_{experiment}": {
                "model": experiment_arg.model_args.model,
                "num_layers": experiment_arg.model_args.num_layers,
                "hidden_channels": experiment_arg.model_args.hidden_channels,
                "num_neighbors": experiment_arg.model_args.num_neighbors,
                "batch_size": experiment_arg.model_args.batch_size,
                "dropout": experiment_arg.model_args.dropout,
                "lr": experiment_arg.model_args.lr,
                "epochs": experiment_arg.model_args.epochs,
                "runs": experiment_arg.model_args.runs,
                "gps_heads": experiment_arg.model_args.gps_heads,
                "gps_attn_type": experiment_arg.model_args.gps_attn_type,
                "gps_attn_dropout": experiment_arg.model_args.gps_attn_dropout,
                "task_name": experiment_arg.data_args.task_name,
                "is_regression": experiment_arg.data_args.is_regression,
            }
        }, allow_val_change=True)
        run_gnn_baseline(
            experiment_arg.data_args,
            experiment_arg.model_args,
            root / cast(str, meta_args.weights_directory) / f'{meta_args.target_col}',
            dataset,
        )
    results = load_all_loss_tuples()
    logging.info('Constructing Plots, across models')
    plot_metric_across_models(results)
    logging.info('Constructing Plots, metric per-encoder')
    plot_metric_per_encoder(results)
    logging.info('Constructing Plots, model per-encoder')
    plot_model_per_encoder(results)

    # Finish wandb run
    wandb.finish()
    logging.info("WandB run finished.")


if __name__ == '__main__':
    main()
