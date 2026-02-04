#!/usr/bin/env python
"""Wandb Sweep training script for GNN experiments with DomainRel fixed split."""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, cast

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalDataset
from credipred.encoders.encoder import Encoder
from credipred.encoders.pre_embedding_encoder import TextEmbeddingEncoder
from credipred.encoders.rni_encoding import RNIEncoder
from credipred.gnn.model import Model
from credipred.utils.args import parse_args
from credipred.utils.logger import setup_logging
from credipred.utils.path import get_root_dir
from credipred.utils.seed import seed_everything


def train_epoch(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []

    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        train_mask = batch.train_mask

        if train_mask.sum() == 0:
            continue

        loss = F.l1_loss(preds[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        all_preds.append(preds[train_mask])
        all_targets.append(targets[train_mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss, r2


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
) -> Tuple[float, float]:
    """Evaluate model on given mask."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        mask = getattr(batch, mask_name)

        if mask.sum() == 0:
            continue

        loss = F.l1_loss(preds[mask], targets[mask])
        total_loss += loss.item()
        total_batches += 1
        all_preds.append(preds[mask])
        all_targets.append(targets[mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    return avg_loss, r2


def main():
    """Main sweep training function."""
    # Initialize wandb
    wandb.init()
    config = wandb.config

    root = get_root_dir()
    config_file_path = root / config.config_file
    meta_args, _ = parse_args(config_file_path)

    setup_logging(None)  # Log to console only during sweep
    seed_everything(meta_args.global_seed)

    # Setup encoders
    encoder_classes: Dict[str, Encoder] = {
        'RNI': RNIEncoder(64),
        'PRE': TextEmbeddingEncoder(64),
    }
    encoding_dict: Dict[str, Encoder] = {}
    for index, value in meta_args.encoder_dict.items():
        encoding_dict[index] = encoder_classes[value]

    # Load dataset
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
        fixed_split_dir=meta_args.fixed_split_dir,
    )

    data = dataset[0]
    split_idx = dataset.get_idx_split()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get hyperparameters from sweep config
    num_layers = config.get('num_layers', 3)
    hidden_channels = config.get('hidden_channels', 256)
    dropout = config.get('dropout', 0.1)
    lr = config.get('lr', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    batch_size = config.get('batch_size', 1024)
    model_name = config.get('model', 'GAT')

    # Compute num_neighbors based on num_layers
    num_neighbors = [30] * num_layers

    # Create data loaders
    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Build model
    model = Model(
        model_name=model_name,
        normalization='BatchNorm',
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=128,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    best_metrics = {}
    patience = 20
    patience_counter = 0
    epochs = 100

    for epoch in range(epochs):
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer)
        val_loss, val_r2 = evaluate(model, val_loader, 'valid_mask')
        test_loss, test_r2 = evaluate(model, test_loader, 'test_mask')

        # Log metrics to wandb
        wandb.log(
            {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/r2': train_r2,
                'val/loss': val_loss,
                'val/r2': val_r2,
                'test/loss': test_loss,
                'test/r2': test_r2,
            }
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                'best_val_loss': best_val_loss,
                'best_val_r2': val_r2,
                'best_epoch': epoch,
                'test_loss_at_best_val': test_loss,
                'test_r2_at_best_val': test_r2,
                'train_loss_at_best_val': train_loss,
                'train_r2_at_best_val': train_r2,
            }
            # Log best metrics
            wandb.run.summary['best_val_loss'] = best_val_loss
            wandb.run.summary['best_epoch'] = epoch
            wandb.run.summary['test_loss_at_best_val'] = test_loss
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch}')
            break

    # Save checkpoint with model config, weights, and metrics
    if best_model_state is not None:
        checkpoint = {
            # Model weights
            'model_state_dict': best_model_state,
            # Model config/hyperparameters
            'config': {
                'model': model_name,
                'num_layers': num_layers,
                'hidden_channels': hidden_channels,
                'dropout': dropout,
                'lr': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size,
                'num_neighbors': num_neighbors,
                'in_channels': data.num_features,
                'out_channels': 128,
                'normalization': 'BatchNorm',
            },
            # Metrics
            'metrics': best_metrics,
            # Wandb info
            'wandb_run_id': wandb.run.id,
            'wandb_run_name': wandb.run.name,
        }

        # Create save directory if not exists
        save_dir = root / 'data' / 'weights_domainrel' / 'sweep'
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        save_path = save_dir / f'{wandb.run.id}.pt'
        torch.save(checkpoint, save_path)
        logging.info(f'Saved checkpoint to {save_path}')

        # Log artifact to wandb
        artifact = wandb.Artifact(
            name=f'model-{wandb.run.id}',
            type='model',
            metadata=checkpoint['config'] | checkpoint['metrics'],
        )
        artifact.add_file(str(save_path))
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    main()
