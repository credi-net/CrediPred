import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.temporal_dataset import TemporalBinaryDataset
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.enums import Metric
from credipred.utils.logger import Logger
from credipred.utils.plot import Scoring, plot_avg_loss
from credipred.utils.save import save_loss_results


def train_(
    model: torch.nn.Module,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
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
        print(preds.size())
        preds = preds.squeeze()
        targets = batch.y
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue

        loss = F.nll_loss(preds[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += train_mask.sum().item()
        all_preds.append(preds[train_mask].argmax(dim=-1))
        all_targets.append(targets[train_mask])

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
    total_random_loss = 0
    total_samples = 0
    all_preds = []
    all_mean_preds = []
    all_random_preds = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index).squeeze()
        targets = batch.y
        n = targets.size(0)
        mask = getattr(batch, mask_name)
        if mask.sum() == 0:
            continue
        # MEAN: 0.546
        mean_preds = torch.full((n, 2), -100.0).to(device)
        mean_preds[:, 1] = 0.0  # High logit for class 1
        random_preds = torch.randn((n, 2), np.log(0.5)).to(device)
        loss = F.nll_loss(preds[mask], targets[mask])
        mean_loss = F.nll_loss(mean_preds, targets[mask])
        random_loss = F.nll_loss(random_preds, targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_random_loss += random_loss.item()
        total_samples += mask.sum().item()

        all_preds.append(preds[mask].argmax(dim=-1))
        all_mean_preds.append(mean_preds.argmax(dim=-1))
        all_random_preds.append(random_preds.argmax(dim=-1))
        all_targets.append(targets[mask])

    avg_ce = total_loss / total_samples
    total_mean_loss / total_samples
    total_random_loss / total_samples

    # Calculate accuracy
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    acc = (y_pred == y_true).float().mean().item()
    acc_mean = (torch.cat(all_mean_preds) == y_true).float().mean().item()
    acc_random = (torch.cat(all_random_preds) == y_true).float().mean().item()

    return (avg_ce, acc, acc_mean, acc_random)


def run_binary_class_gnn_baseline(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalBinaryDataset,
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
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_arguments.lr)
        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        for _ in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            loss_ce, _ = train_(model, train_loader, optimizer)
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
    logging.info(logger.get_statistics(metric=Metric.acc))
    logging.info(logger.get_avg_statistics(metric=Metric.acc))
    logging.info('Constructing plots')
    plot_avg_loss(
        loss_tuple_run_mse, model_arguments.model, Scoring.mae, 'loss_plot.png'
    )
    logging.info('Saving pkl of results')
    save_loss_results(
        loss_tuple_run_mse, model_arguments.model, 'binary_classification'
    )
