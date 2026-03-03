import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import (
    TemporalBinaryDatasetAllMultiGlobalSplits,
)
from credipred.gnn.model import Model, ModelFlags
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.enums import Metric, TrainingMethods
from credipred.utils.logger import Logger
from credipred.utils.plot import Scoring, plot_avg_loss
from credipred.utils.save import save_loss_results


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
    training_method: TrainingMethods,
    alpha: float = 0.5,
) -> Tuple[float, float]:
    model.train()
    device = next(model.parameters()).device
    total_loss = 0
    total_samples = 0
    all_preds_cls = []
    all_targets_cls = []
    for batch in tqdm(train_loader, desc='Batchs', leave=False):
        optimizer.zero_grad()
        batch = batch.to(device)
        preds_cls, preds_reg = model(batch.x, batch.edge_index)
        preds_reg = preds_reg.squeeze(0)
        targets_cls = batch.y[:0]
        targets_reg = batch.y[:1]
        active_mask = batch.train_mask[: batch.size]

        mask_bin = targets_cls != -1
        mask_reg = targets_reg != -1.0

        if not mask_bin.any() and not mask_reg.any():
            continue

        loss_cls = torch.tensor(0.0, device=device)
        if mask_bin.any():
            batch_weights = None
            if training_method == TrainingMethods.WEIGHTED_LOSS:
                num_pos = (targets_cls[mask_bin] == 1).sum().float()
                num_neg = (targets_cls[mask_bin] == 0).sum().float()

                if num_pos > 0 and num_neg > 0:
                    batch_weights = torch.tensor(
                        [1.0, num_neg / num_pos], device=device
                    )

            loss_cls = F.cross_entropy(
                preds_cls[mask_bin], targets_cls[mask_bin], weight=batch_weights
            )

        loss_reg = torch.tensor(0.0, device=device)

        if mask_reg.any():
            loss_reg = F.l1_loss(preds_reg[mask_reg], targets_reg[mask_reg])

        loss = (alpha * loss_cls) + ((1 - alpha) * loss_reg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += 1

        if mask_bin.any():
            all_preds_cls.append(preds_cls[mask_bin].argmax(dim=-1))
            all_targets_cls.append(targets_cls[mask_bin])

    avg_loss = total_loss / total_samples
    # Calculate accuracy
    acc = 0.0
    if all_preds_cls:
        y_pred = torch.cat(all_preds_cls)
        y_true = torch.cat(all_targets_cls)
        acc = (y_pred == y_true).float().mean().item()
    return (avg_loss, acc)


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
        preds_cls, preds_reg = model(batch.x, batch.edge_index)
        preds_cls = preds_cls.squeeze()
        targets_cls = batch.y[:, 0]
        targets_reg = batch.y[:, 1]
        mask = getattr(batch, mask_name)
        n = targets_cls.size(0)
        if mask.sum() == 0:
            continue
        # MEAN: 0.546
        mean_preds = torch.full((n, 2), -100.0).to(device)
        mean_preds[:, 1] = 0.0  # High logit for class 1
        loss = F.nll_loss(preds_cls[mask], targets_cls[mask])
        mean_loss = F.nll_loss(mean_preds[mask], targets_cls[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_samples += mask.sum().item()

        all_preds.append(preds_cls[mask].argmax(dim=-1))
        all_mean_preds.append(mean_preds[mask].argmax(dim=-1))
        all_targets.append(targets_cls[mask])

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


def run_multihead_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalBinaryDatasetAllMultiGlobalSplits,
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
    global_best_val_loss = float('inf')
    best_state_dict = None
    logging.info('*** Training ***')
    flags = ModelFlags(binary=False, multi_head=True)
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            flags=flags,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
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
            if valid_ce_loss < global_best_val_loss:
                global_best_val_loss = valid_ce_loss
                best_state_dict = model.state_dict()

        loss_tuple_run_mse.append(loss_tuple_epoch_mse)

    best_model_dir = weight_directory / f'{model_arguments.model}'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = best_model_dir / 'best_model.pt'
    torch.save(best_state_dict, best_model_path)
    logging.info(f'Model: {model_arguments} weights saved to: {best_model_path}')
    logging.info('*** Statistics ***')
    logging.info(logger.get_statistics(metric=Metric.acc, higher_is_better=True))
    logging.info(logger.get_avg_statistics(metric=Metric.acc, higher_is_better=True))
    logging.info('Constructing plots')
    plot_avg_loss(
        loss_tuple_run_mse, model_arguments.model, Scoring.acc, 'loss_plot.png'
    )
    logging.info('Saving pkl of results')
    save_loss_results(
        loss_tuple_run_mse, model_arguments.model, 'binary_classification'
    )
