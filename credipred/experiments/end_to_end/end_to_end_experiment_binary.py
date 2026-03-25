import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from credipred.dataset.dataset import WebGraphDataset
from credipred.gnn.model import Model
from credipred.head.decoder import LabelPredictor
from credipred.utils.args import DataArguments, ModelArguments
from credipred.utils.domain_handler import reverse_domain
from credipred.utils.enums import Metric, TrainingMethods
from credipred.utils.logger import Logger
from credipred.utils.plot import Scoring, plot_avg_loss
from credipred.utils.save import save_loss_results

embedding_dict_cache: OrderedDict[str, Dict] = OrderedDict()
idx_to_domain: Dict = dict()


def get_text_embeddings(
    embeddings_lookup_table: Dict[str, str],
    embedding_location: Path,
    seed_nodes: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    n = len(seed_nodes)
    out = torch.empty((n, 64), dtype=torch.float32, device=device)
    for i, node_idx in enumerate(seed_nodes):
        name = reverse_domain(idx_to_domain[node_idx.item()])
        if name in embeddings_lookup_table:
            wet_file_name = embeddings_lookup_table[name]
            if wet_file_name in embedding_dict_cache:
                embedding_dict_cache.move_to_end(wet_file_name)
            else:
                path = embedding_location / (wet_file_name + '.pkl')
                with open(path, 'rb') as file:
                    logging.info('Pushing to embedding dictionary.')
                    embedding_dict_cache[wet_file_name] = pickle.load(file)

            entries = embedding_dict_cache[wet_file_name][name]
            embeddings = [e[1] for e in entries if len(e) == 2]
            stacked_embs = torch.tensor(np.array(embeddings), dtype=torch.float32)
            aggregated_emb = torch.mean(stacked_embs, dim=0)
            out[i] = aggregated_emb[0:64]
        else:
            out[i] = torch.rand(64, dtype=torch.float32)

    return out


def train_(
    model: torch.nn.ModuleList,
    train_loader: NeighborLoader,
    optimizer: torch.optim.AdamW,
    training_method: TrainingMethods,
    embeddings_location: Path,
    embeddings_lookup_table: Dict[str, str],
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
        preds = model[0].get_embeddings(batch.x, batch.edge_index)
        # Only compute loss on seed nodes (first batch_size nodes).
        n_seed = batch.batch_size
        seed_nodes = batch.n_id[:n_seed]
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
                seed_nodes = seed_nodes[active_mask]
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

        seed_text_embeddings = get_text_embeddings(
            embeddings_lookup_table, embeddings_location, seed_nodes, device
        )

        pred_text_gnn_embeddings = torch.cat(
            (seed_preds, seed_text_embeddings), dim=1
        )  # Dimension one: horizontal concatenation.

        predictions = model[1](pred_text_gnn_embeddings)
        loss = F.nll_loss(predictions, seed_targets, weight=batch_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_samples += seed_targets.size(0)
        all_preds.append(predictions.argmax(dim=-1))
        all_targets.append(seed_targets)

    avg_ce = total_loss / total_samples
    # Calculate accuracy
    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)
    acc = (y_pred == y_true).float().mean().item()
    return (avg_ce, acc)


@torch.no_grad()
def evaluate(
    model: torch.nn.ModuleList,
    loader: NeighborLoader,
    mask_name: str,
    embeddings_location: Path,
    embeddings_lookup_table: Dict[str, str],
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
        preds = model[0].get_embeddings(batch.x, batch.edge_index)
        targets = batch.y
        # Only evaluate seed nodes (first batch_size nodes) to avoid
        # double-counting nodes that appear as neighbors in other batches.
        n_seed = batch.batch_size
        mask = getattr(batch, mask_name)[:n_seed]
        if mask.sum() == 0:
            continue
        seed_preds = preds[:n_seed]
        seed_nodes = batch.n_id[:n_seed]
        seed_targets = targets[:n_seed]
        # MEAN: 0.546
        mean_preds = torch.full((n_seed, 2), -100.0).to(device)
        mean_preds[:, 1] = 0.0  # High logit for class 1
        seed_text_embeddings = get_text_embeddings(
            embeddings_lookup_table, embeddings_location, seed_nodes, device
        )
        pred_text_gnn_embeddings = torch.cat(
            (seed_preds, seed_text_embeddings), dim=1
        )  # Dimension one: horizontal concatenation.
        predictions = model[1](pred_text_gnn_embeddings)
        loss = F.nll_loss(predictions[mask], seed_targets[mask])
        mean_loss = F.nll_loss(mean_preds[mask], seed_targets[mask])

        total_loss += loss.item()
        total_mean_loss += mean_loss.item()
        total_samples += mask.sum().item()

        all_preds.append(predictions[mask].argmax(dim=-1))
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


def run_end_to_end_binary_classification(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: WebGraphDataset,
    embeddings_location: Path,
    embeddings_lookup_table: Dict[str, str],
) -> None:
    data = dataset[0]
    domain_to_idx_mapping = dataset.get_mapping()
    global idx_to_domain
    idx_to_domain = {v: k for k, v in domain_to_idx_mapping.items()}
    logging.info('idx to domain mapping completed.')
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
    patience = model_arguments.patience
    patience_counter = 0
    logging.info('*** Training ***')
    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        gnn_model = Model(
            model_name=model_arguments.model,
            normalization=model_arguments.normalization,
            in_channels=data.num_features,
            hidden_channels=model_arguments.hidden_channels,
            out_channels=model_arguments.embedding_dimension,
            num_layers=model_arguments.num_layers,
            dropout=model_arguments.dropout,
            binary=True,
        ).to(device)
        mlp_model = LabelPredictor(in_dim=(model_arguments.hidden_channels + 64)).to(
            device
        )
        model = torch.nn.ModuleList([gnn_model, mlp_model])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model_arguments.lr,
        )
        loss_tuple_epoch_mse: List[Tuple[float, float, float, float, float]] = []
        for epoch in tqdm(range(1, 1 + model_arguments.epochs), desc='Epochs'):
            loss_ce, _ = train_(
                model,
                train_loader,
                optimizer,
                model_arguments.training_method,
                embeddings_location,
                embeddings_lookup_table,
            )
            train_ce_loss, train_acc, _, _ = evaluate(
                model,
                train_loader,
                'train_mask',
                embeddings_location,
                embeddings_lookup_table,
            )
            valid_ce_loss, valid_acc, valid_mean_acc, _ = evaluate(
                model,
                val_loader,
                'valid_mask',
                embeddings_location,
                embeddings_lookup_table,
            )
            (
                test_ce_loss,
                test_acc,
                test_mean_acc,
                test_random_acc,
            ) = evaluate(
                model,
                test_loader,
                'test_mask',
                embeddings_location,
                embeddings_lookup_table,
            )
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
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f'Early stopping at epoch {epoch}')
                    logging.info(f'Best validation loss {global_best_val_loss}')
                    break

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
