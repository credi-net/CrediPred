import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.dataset import WebGraphDataset
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.enums import Metric, TrainingMethods
from credipred.utils.logger import Logger


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
    training_method: TrainingMethods,
) -> Tuple[float, float]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        # Only compute loss on seed nodes (first batch_size nodes).
        n_seed = batch.batch_size
        seed_preds = preds[:n_seed]
        seed_targets = batch.y[:n_seed]
        batch_weights = None

        match training_method:
            case TrainingMethods.DEFAULT:
                continue
            case TrainingMethods.DOWN_SAMPLE:
                pos_idx = torch.where(seed_targets == 1)[0]
                neg_idx = torch.where(seed_targets == 0)[0]

                n_pos = pos_idx.numel()
                n_neg = neg_idx.numel()
                if n_pos == 0 or n_neg == 0:
                    continue

                if n_pos > n_neg:
                    perm = torch.randperm(n_pos, device=device)[:n_neg]
                    pos_idx = pos_idx[perm]

                active_mask = torch.zeros(n_seed, dtype=torch.bool, device=device)
                active_mask[pos_idx] = True
                active_mask[neg_idx] = True

                seed_preds = seed_preds[active_mask]
                seed_targets = seed_targets[active_mask]

            case TrainingMethods.WEIGHTED_LOSS:
                num_pos = (seed_targets == 1).sum().float()
                num_neg = (seed_targets == 0).sum().float()

                if num_pos > 0 and num_neg > 0:
                    weight_neg = 1.0
                    weight_pos = num_neg / num_pos
                    batch_weights = torch.tensor(
                        [weight_neg, weight_pos], device=device
                    )
                else:
                    batch_weights = None

        loss = F.nll_loss(seed_preds, seed_targets, weight=batch_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += seed_targets.size(0)
        all_preds.append(seed_preds.argmax(dim=-1))
        all_targets.append(seed_targets)

    avg_ce = total_loss / total_samples
    # Calculate accuracy
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    acc = (y_pred == y_true).float().mean().item()
    return (avg_ce, acc)


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
    total_samples = 0
    all_preds = []
    all_mean_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)
        targets = batch.y
        # Only evaluate seed nodes (first batch_size nodes) to avoid
        # double-counting nodes that appear as neighbors in other batches.
        n_seed = batch.batch_size
        mask = getattr(batch, mask_name)[:n_seed]
        if mask.sum() == 0:
            continue
        seed_preds = preds[:n_seed]
        seed_targets = targets[:n_seed]
        # MEAN: 0.546
        mean_preds = torch.full((n_seed, 2), -100.0).to(device)
        mean_preds[:, 1] = 0.0  # High logit for class 1
        loss = F.nll_loss(seed_preds[mask], seed_targets[mask])
        mean_loss = F.nll_loss(mean_preds[mask], seed_targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_samples += mask.sum().item()

        all_preds.append(seed_preds[mask].argmax(dim=-1))
        all_mean_preds.append(mean_preds[mask].argmax(dim=-1))
        all_targets.append(seed_targets[mask])

    avg_ce = total_loss / total_samples
    total_mean_loss / total_samples

    # Calculate accuracy
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    acc = (y_pred == y_true).float().mean().item()
    acc_mean = (torch.cat(all_mean_preds) == y_true).float().mean().item()

    random_choices = torch.randint(0, 2, (y_true.size(0),), device=device)
    acc_random = (random_choices == y_true).float().mean().item()

    return (avg_ce, acc, acc_mean, acc_random)


def run_binary_class_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: WebGraphDataset,
) -> None:
    data = dataset[0].cpu()
    split_idx = dataset.get_idx_split()
    logging.info(
        'Setting up training for task of: %s on model: %s',
        data_arguments.task_name,
        model_arguments.model,
    )
    device = f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logging.info(f'Device found: {device}')

    logging.info(f'Dataset features on device: {data.x.device}')
    logging.info(f'Dataset Edge Index on device: {data.edge_index.device}')

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
        drop_last=True,
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
        drop_last=True,
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
        drop_last=True,
    )
    logging.info('Test loader created')

    logger = Logger(model_arguments.runs)
    loss_tuple_run_mse: List[List[Tuple[float, float, float, float, float]]] = []
    global_best_val_loss = float('inf')
    best_state_dict = None
    patience = model_arguments.patience
    patience_counter = 0
    logging.info('*** Training ***')
    if model_arguments.model == 'GPS':
        kwargs = {
            'gps_head': 1,
            'gps_attn_type': 'performer',
            'gps_local_mpnn': 'gatedgcn',
        }
    else:
        kwargs = {}
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            binary=True,
            kwargs=kwargs,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        best_val_per_epoch = float('inf')
        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            loss_ce, _ = train_(
                model, train_loader, optimizer, model_arguments.training_method
            )
            train_ce_loss, train_acc, _, _ = evaluate(model, train_loader, 'train_mask')
            valid_ce_loss, valid_acc, valid_mean_acc, _ = evaluate(
                model, val_loader, 'valid_mask'
            )
            (
                test_ce_loss,
                test_acc,
                test_mean_acc,
                test_random_acc,
            ) = evaluate(model, test_loader, 'test_mask')
            result = (
                train_acc,
                valid_acc,
                test_acc,
                test_mean_acc,
                test_random_acc,
            )
            loss_tuple_epoch_mse.append(result)
            logger.add_result(
                run,
                (
                    train_acc,
                    valid_acc,
                    test_acc,
                    valid_mean_acc,
                    test_random_acc,
                ),
            )
            if valid_ce_loss < best_val_per_epoch:
                best_val_per_epoch = valid_ce_loss
                best_state_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Early stopping at epoch {epoch}')
                    logging.info(f'Best validation loss {global_best_val_loss}')
                    break

        if best_val_per_epoch < global_best_val_loss:
            global_best_val_loss = best_val_per_epoch
            best_state_dict = model.state_dict()

        loss_tuple_run_mse.append(loss_tuple_epoch_mse)

    best_model_dir = weight_directory / f'{model_arguments.model}'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / 'best_model.pt'
    torch.save(best_state_dict, best_model_path)
    logging.info(f'Model: {model_arguments} weights saved to: {best_model_path}')
    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics(metric=Metric.acc, higher_is_better=True))
