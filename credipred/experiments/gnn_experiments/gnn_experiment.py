import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch_geometric.loader import NeighborLoader
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalDataset
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.logger import Logger
from credipred.utils.plot import (
    Scoring,
    mean_across_lists,
    plot_avg_loss,
    plot_avg_loss_r2,
    plot_pred_target_distributions_bin_list,
)
from credipred.utils.prob import ragged_mean_by_index
from credipred.utils.save import save_loss_results


def train(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
) -> Tuple[float, float, Tensor, Tensor]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        # For GPS on a single large graph, use batch=None to enable global attention
        # across ALL sampled nodes (not just within each seed's neighborhood)
        preds = model(batch.x, batch.edge_index, batch=None).squeeze()
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
    avg_preds = ragged_mean_by_index(all_preds)
    avg_targets = ragged_mean_by_index(all_targets)
    mse = total_loss / total_batches
    return (mse, r2, avg_preds, avg_targets)


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
) -> Tuple[float, float, List[float], List[float]]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    # TODO: Score in one list
    pred_scores = []
    target_scores = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        # For GPS on a single large graph, use batch=None to enable global attention
        # across ALL sampled nodes (not just within each seed's neighborhood)
        preds = model(batch.x, batch.edge_index, batch=None).squeeze()
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
        for pred in preds[train_mask]:
            pred_scores.append(pred.item())
        for targ in targets[train_mask]:
            target_scores.append(targ.item())

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    ragged_mean_by_index(all_preds)
    ragged_mean_by_index(all_targets)
    mse = total_loss / total_batches
    return (mse, r2, pred_scores, target_scores)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: NeighborLoader,
    mask_name: str,
) -> Tuple[float, float, float, float]:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0
    total_mean_loss = 0
    total_random_loss = 0
    total_batches = 0
    all_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        # For GPS on a single large graph, use batch=None to enable global attention
        # across ALL sampled nodes (not just within each seed's neighborhood)
        preds = model(batch.x, batch.edge_index, batch=None).squeeze()
        targets = batch.y
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        # MEAN: 0.546
        mean_preds = torch.full(batch.y[mask].size(), 0.5).to(device)
        random_preds = torch.rand(batch.y[mask].size(0)).to(device)
        loss = F.l1_loss(preds[mask], targets[mask])
        mean_loss = F.l1_loss(mean_preds, targets[mask])
        random_loss = F.l1_loss(random_preds, targets[mask])

        # TODO: Change this to report the loss of mean to be accurate. Use full score for don't average per batch.
        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        total_batches += 1

        all_preds.append(preds[mask])
        all_targets.append(targets[mask])

    r2 = r2_score(torch.cat(all_preds), torch.cat(all_targets)).item()
    mse = total_loss / total_batches
    mse_mean = total_mean_loss / total_batches
    mse_random = total_random_loss / total_batches
    return (mse, mse_mean, mse_random, r2)


def run_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalDataset,
) -> None:
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logging.info(f'Device found: {device}')

    logging.info(f'Training set size: {split_idx["train"].size()}')
    logging.info(f'Validation set size: {split_idx["valid"].size()}')
    logging.info(f'Testing set size: {split_idx["test"].size()}')

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Train loader created')

    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    logging.info('Valid loader created')
    test_loader = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    logging.info('Test loader created')

    logger = Logger(model_arguments.runs)
    loss_tuple_run_mse: List[List[Tuple[float, float, float, float, float]]] = []
    loss_tuple_run_r2: List[List[Tuple[float, float, float]]] = []
    final_avg_preds: List[List[float]] = []
    final_avg_targets: List[List[float]] = []
    global_best_val_loss = float('inf')
    best_state_dict = None
    logging.info('*** Training ***')

    # Get current learning rate for logging
    current_lr = model_arguments.lr
    global_step = 0

    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        # Build GPS-specific attention kwargs if using GPS
        gps_attn_kwargs = None
        if model_arguments.model == 'GPS':
            gps_attn_kwargs = {'dropout': model_arguments.gps_attn_dropout}

        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            # GraphGPS specific parameters
            gps_heads=model_arguments.gps_heads,
            gps_attn_type=model_arguments.gps_attn_type,
            gps_attn_kwargs=gps_attn_kwargs,
            gps_local_mpnn=model_arguments.gps_local_mpnn,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_arguments.lr,
            weight_decay=model_arguments.weight_decay,
        )

        # Log model architecture to wandb (only on first run)
        if run == 0:
            wandb.watch(model, log='all', log_freq=100)

        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        loss_tuple_epoch_r2: List[Tuple[float, float, float]] = []
        epoch_avg_preds: List[List[float]] = []
        epoch_avg_targets: List[List[float]] = []

        # Early stopping variables
        best_val_loss_in_run = float('inf')
        epochs_without_improvement = 0

        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            _, _, batch_preds, batch_targets = train_(model, train_loader, optimizer)
            epoch_avg_preds.append(batch_preds)
            epoch_avg_targets.append(batch_targets)
            train_loss, _, _, train_r2 = evaluate(model, train_loader, 'train_mask')
            valid_loss, valid_mean_baseline_loss, _, valid_r2 = evaluate(
                model, val_loader, 'valid_mask'
            )
            test_loss, test_mean_baseline_loss, test_random_baseline_loss, test_r2 = (
                evaluate(model, test_loader, 'test_mask')
            )

            # Log metrics to wandb
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']

            # GPU memory tracking
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0
            gpu_utilization = 0
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1e9  # GB
                gpu_max_memory = torch.cuda.max_memory_allocated(device) / 1e9  # GB

            wandb.log(
                {
                    # Training metrics
                    'train/loss': train_loss,
                    'train/r2': train_r2,
                    # Validation metrics
                    'val/loss': valid_loss,
                    'val/r2': valid_r2,
                    'val/mean_baseline_loss': valid_mean_baseline_loss,
                    # Test metrics
                    'test/loss': test_loss,
                    'test/r2': test_r2,
                    'test/mean_baseline_loss': test_mean_baseline_loss,
                    'test/random_baseline_loss': test_random_baseline_loss,
                    # Learning rate
                    'train/learning_rate': current_lr,
                    # GPU metrics
                    'gpu/memory_allocated_gb': gpu_memory_allocated,
                    'gpu/memory_reserved_gb': gpu_memory_reserved,
                    'gpu/max_memory_allocated_gb': gpu_max_memory
                    if torch.cuda.is_available()
                    else 0,
                    # Progress tracking
                    'progress/run': run,
                    'progress/epoch': epoch,
                    'progress/global_step': global_step,
                    # Best validation loss tracking
                    'best/val_loss': global_best_val_loss
                    if valid_loss >= global_best_val_loss
                    else valid_loss,
                }
            )

            result = (
                train_loss,
                valid_loss,
                test_loss,
                test_mean_baseline_loss,
                test_random_baseline_loss,
            )
            result_r2 = (train_r2, valid_r2, test_r2)
            loss_tuple_epoch_mse.append(result)
            loss_tuple_epoch_r2.append(result_r2)
            logger.add_result(
                run, (train_loss, valid_loss, test_loss, valid_mean_baseline_loss)
            )
            if valid_loss < global_best_val_loss:
                global_best_val_loss = valid_loss
                best_state_dict = model.state_dict()
                # Log best model checkpoint info
                wandb.run.summary['best_val_loss'] = global_best_val_loss
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_run'] = run

            # Early stopping logic
            if valid_loss < best_val_loss_in_run:
                best_val_loss_in_run = valid_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                model_arguments.patience > 0
                and epochs_without_improvement >= model_arguments.patience
            ):
                logging.info(
                    f'Early stopping at epoch {epoch}. '
                    f'No improvement for {model_arguments.patience} epochs. '
                    f'Best val loss in run: {best_val_loss_in_run:.4f}'
                )
                break

        final_avg_preds.append(mean_across_lists(epoch_avg_preds))
        final_avg_targets.append(mean_across_lists(epoch_avg_targets))
        loss_tuple_run_mse.append(loss_tuple_epoch_mse)
        loss_tuple_run_r2.append(loss_tuple_epoch_r2)

    best_model_dir = weight_directory / f'{model_arguments.model}'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / 'best_model.pt'
    torch.save(best_state_dict, best_model_path)
    logging.info(f'Model: {model_arguments} weights saved to: {best_model_path}')

    # Save model as wandb artifact
    model_artifact = wandb.Artifact(
        name=f'{model_arguments.model}-best-model',
        type='model',
        description=f'Best model checkpoint for {model_arguments.model}',
    )
    model_artifact.add_file(str(best_model_path))
    wandb.log_artifact(model_artifact)

    logging.info('*** Statistics ***')
    statistics = logger.get_statistics()
    avg_statistics = logger.get_avg_statistics()
    logging.info(statistics)
    logging.info(avg_statistics)

    # Log final statistics to wandb
    error_10 = logger.per_run_within_error(
        preds=final_avg_preds, targets=final_avg_targets, percent=10
    )
    error_5 = logger.per_run_within_error(
        preds=final_avg_preds, targets=final_avg_targets, percent=5
    )
    error_1 = logger.per_run_within_error(
        preds=final_avg_preds, targets=final_avg_targets, percent=1
    )

    logging.info(error_10)
    logging.info(error_5)
    logging.info(error_1)

    # Log summary statistics to wandb
    wandb.run.summary.update(
        {
            'final/model': model_arguments.model,
            'final/best_val_loss': global_best_val_loss,
            'final/within_10pct_error': error_10,
            'final/within_5pct_error': error_5,
            'final/within_1pct_error': error_1,
            'final/total_epochs': model_arguments.epochs * model_arguments.runs,
            'final/total_runs': model_arguments.runs,
        }
    )
    logging.info('Constructing plots')
    plot_pred_target_distributions_bin_list(
        preds=final_avg_preds,
        targets=final_avg_targets,
        model_name=model_arguments.model,
        bins=100,
    )
    plot_avg_loss(
        loss_tuple_run_mse, model_arguments.model, Scoring.mae, 'loss_plot.png'
    )
    plot_avg_loss_r2(
        loss_tuple_run_r2, model_arguments.model, Scoring.r2, 'r2_plot.png'
    )
    logging.info('Saving pkl of results')
    save_loss_results(loss_tuple_run_mse, model_arguments.model, 'TODO')
