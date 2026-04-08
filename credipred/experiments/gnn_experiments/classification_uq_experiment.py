"""Classification uncertainty quantification via conformal prediction + ConfGNN.

Three-stage pipeline following the original CF-GNN (snap-stanford/conformalized-gnn):
1. Load pre-trained base classification model, extract softmax probabilities.
2. Evaluate vanilla conformal prediction (APS/RAPS) on base model.
3. Train a ConfGNN correction model with two-phase loss:
   - Phase 1 (first `pred_only_epochs` epochs): NLL prediction loss only.
   - Phase 2 (remaining epochs): NLL + size loss (encourages smaller prediction sets).
4. Evaluate conformal prediction on corrected model, compare with base.

No structure learning — just topology-aware correction.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from credipred.conformal_classification.conf_gnn import ConfGNN
from credipred.conformal_classification.conformal import run_conformal_eval
from credipred.dataset.temporal_dataset import TemporalBinaryDatasetAllGlobalSplits
from credipred.gnn.model import Model
from credipred.utils.args import DataArguments, ModelArguments


# ---------------------------------------------------------------------------
# Size loss (from original CF-GNN)
# ---------------------------------------------------------------------------

def _size_loss(
    softmax_preds: torch.Tensor,
    qhat: float,
    tau: float = 0.5,
    target_size: float = 1.0,
) -> torch.Tensor:
    """Differentiable size loss encouraging smaller prediction sets.

    size_loss = mean(relu(sum(sigmoid((softmax - qhat) / tau)) - target_size))
    """
    soft_membership = torch.sigmoid((softmax_preds - qhat) / tau)
    set_sizes = soft_membership.sum(dim=1)
    return torch.relu(set_sizes - target_size).mean()


# ---------------------------------------------------------------------------
# Extract base predictions
# ---------------------------------------------------------------------------

def _extract_base_softmax(
    data,
    model_arguments: ModelArguments,
    weight_directory: Path,
    device: torch.device,
) -> torch.Tensor:
    """Load base classification model and extract softmax for all nodes."""
    logging.info('=== Loading base classification model ===')
    base_model = Model(
        model_name=model_arguments.model,
        normalization=model_arguments.normalization,
        in_channels=data.num_features,
        hidden_channels=model_arguments.hidden_channels,
        out_channels=model_arguments.embedding_dimension,
        num_layers=model_arguments.num_layers,
        dropout=model_arguments.dropout,
        binary=True,
    ).to(device)

    base_path = weight_directory / model_arguments.model / 'best_model.pt'
    if not base_path.exists():
        raise FileNotFoundError(f'Base classification model not found at {base_path}')
    state_dict = torch.load(base_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    logging.info('Loaded base model from %s', base_path)

    # Full-batch inference: base model outputs log_softmax (LabelPredictor)
    # For large graphs we use NeighborLoader to avoid OOM
    from torch_geometric.loader import NeighborLoader

    all_softmax = torch.zeros(data.num_nodes, 2, device='cpu')
    extract_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )

    with torch.no_grad():
        for batch in tqdm(extract_loader, desc='Extracting base softmax'):
            batch = batch.to(device)
            log_probs = base_model(batch.x, batch.edge_index)  # log_softmax
            probs = log_probs.exp()  # convert to softmax probabilities
            n_seed = batch.batch_size
            original_ids = batch.n_id[:n_seed]
            all_softmax[original_ids] = probs[:n_seed].cpu()

    logging.info(
        'Base softmax stats — class 0: mean=%.4f, class 1: mean=%.4f',
        all_softmax[:, 0].mean(), all_softmax[:, 1].mean(),
    )

    del base_model
    torch.cuda.empty_cache()
    return all_softmax


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_classification_uq(
    data_arguments: DataArguments,
    model_arguments: ModelArguments,
    weight_directory: Path,
    dataset: TemporalBinaryDatasetAllGlobalSplits,
) -> None:
    """Classification UQ via conformal prediction + ConfGNN correction."""
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    device = torch.device(
        f'cuda:{model_arguments.device}' if torch.cuda.is_available() else 'cpu',
    )

    alpha = model_arguments.conformal_alpha
    confgnn_epochs = model_arguments.confgnn_epochs
    pred_only_epochs = model_arguments.confgnn_pred_only_epochs
    tau = model_arguments.confgnn_tau
    target_size = model_arguments.confgnn_target_size
    size_loss_weight = model_arguments.confgnn_size_loss_weight
    confgnn_lr = model_arguments.confgnn_lr

    logging.info('Setting up classification UQ for model: %s', model_arguments.model)
    logging.info('Device: %s', device)
    logging.info('Conformal alpha: %.3f', alpha)
    logging.info('ConfGNN epochs: %d (pred-only: %d)', confgnn_epochs, pred_only_epochs)
    logging.info('Train: %d, Valid: %d, Test: %d',
                 split_idx['train'].size(0), split_idx['valid'].size(0), split_idx['test'].size(0))

    # ---- Stage 1: Extract base softmax (with cache) ----
    cache_path = weight_directory / model_arguments.model / 'base_softmax.pt'
    if cache_path.exists():
        logging.info('Loading cached base softmax from %s', cache_path)
        base_softmax = torch.load(cache_path, map_location='cpu', weights_only=True)
    else:
        base_softmax = _extract_base_softmax(data, model_arguments, weight_directory, device)
        torch.save(base_softmax, cache_path)
        logging.info('Saved base softmax cache to %s', cache_path)

    val_idx = split_idx['valid'].numpy()
    test_idx = split_idx['test'].numpy()
    labels_np = data.y.cpu().numpy()
    base_smx_np = base_softmax.numpy()

    # ---- Stage 2: Vanilla conformal evaluation ----
    logging.info('=== Vanilla Conformal Prediction (Base Model) ===')
    for score in ['aps', 'raps']:
        cov, eff, qhat = run_conformal_eval(
            base_smx_np, labels_np, val_idx, test_idx, alpha, score=score,
        )
        logging.info(
            '  %s: coverage=%.4f efficiency=%.4f qhat=%.4f',
            score.upper(), cov, eff, qhat,
        )
        wandb.log({
            f'base/{score}_coverage': cov,
            f'base/{score}_efficiency': eff,
            f'base/{score}_qhat': qhat,
        })

    # Get qhat from APS on validation set for size loss
    # Use a single split (not 100 trials) for the training qhat
    n_val = len(val_idx)
    n_cal = n_val // 2
    np.random.seed(0)
    perm = np.random.permutation(n_val)
    cal_local = val_idx[perm[:n_cal]]
    eval_local = val_idx[perm[n_cal:]]
    from credipred.conformal_classification.conformal import aps as aps_fn
    _, _, _, qhat_for_size = aps_fn(
        base_smx_np[cal_local], base_smx_np[eval_local],
        labels_np[cal_local], labels_np[eval_local], alpha,
    )
    logging.info('qhat for size loss: %.4f', qhat_for_size)

    # ---- Stage 3: Train ConfGNN (mini-batch with NeighborLoader) ----
    logging.info('=== Training ConfGNN Correction ===')

    from torch_geometric.loader import NeighborLoader

    # Replace node features with base_softmax for ConfGNN
    data.x = base_softmax

    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )

    best_val_loss = float('inf')
    best_state_dict = None

    for run in tqdm(range(model_arguments.runs), desc='Runs'):
        conf_model = ConfGNN(
            num_classes=2,
            hidden_channels=64,
            num_layers=2,
            dropout=model_arguments.dropout,
            normalization=model_arguments.normalization,
        ).to(device)
        optimizer = torch.optim.AdamW(
            conf_model.parameters(),
            lr=confgnn_lr,
            weight_decay=model_arguments.weight_decay,
        )

        for epoch in tqdm(range(1, 1 + confgnn_epochs), desc=f'Run {run+1} Epochs'):
            conf_model.train()
            epoch_loss = 0.0
            epoch_size_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = batch.to(device)
                n_seed = batch.batch_size
                optimizer.zero_grad()

                delta = conf_model(batch.x, batch.edge_index)
                corrected = batch.x + delta
                corrected_log_probs = F.log_softmax(corrected, dim=1)

                # Prediction loss (NLL) on seed nodes only
                pred_loss = F.nll_loss(
                    corrected_log_probs[:n_seed], batch.y[:n_seed],
                )

                # Size loss (phase 2 only)
                if epoch > pred_only_epochs:
                    corrected_smx = F.softmax(corrected[:n_seed], dim=1)
                    s_loss = _size_loss(corrected_smx, qhat_for_size, tau, target_size)
                    loss = pred_loss + size_loss_weight * s_loss
                else:
                    s_loss = torch.tensor(0.0)
                    loss = pred_loss

                loss.backward()
                optimizer.step()
                epoch_loss += pred_loss.item()
                epoch_size_loss += s_loss.item()
                n_batches += 1

            # Evaluate on validation
            if epoch % model_arguments.log_steps == 0 or epoch == confgnn_epochs:
                conf_model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                train_acc_sum = 0.0
                train_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        n_seed = batch.batch_size
                        delta_eval = conf_model(batch.x, batch.edge_index)
                        corrected_eval = batch.x + delta_eval
                        corr_log_probs = F.log_softmax(corrected_eval, dim=1)
                        val_loss_sum += F.nll_loss(
                            corr_log_probs[:n_seed], batch.y[:n_seed], reduction='sum',
                        ).item()
                        val_correct += (corr_log_probs[:n_seed].argmax(1) == batch.y[:n_seed]).sum().item()
                        val_total += n_seed

                val_loss = val_loss_sum / val_total
                val_acc = val_correct / val_total
                avg_train_loss = epoch_loss / max(n_batches, 1)
                avg_size_loss = epoch_size_loss / max(n_batches, 1)

                logging.info(
                    'Run %d Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f size_loss=%.4f',
                    run + 1, epoch, avg_train_loss, val_loss, val_acc, avg_size_loss,
                )
                wandb.log({
                    'confgnn/train_loss': avg_train_loss,
                    'confgnn/val_loss': val_loss,
                    'confgnn/val_acc': val_acc,
                    'confgnn/size_loss': avg_size_loss,
                    'confgnn/epoch': epoch,
                    'confgnn/run': run,
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = conf_model.state_dict().copy()

    # Save best model
    save_dir = weight_directory / model_arguments.model
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'conf_gnn_model.pt'
    torch.save(best_state_dict, save_path)
    logging.info('Saved ConfGNN to %s', save_path)

    # ---- Stage 4: Evaluate corrected model ----
    logging.info('=== Conformal Prediction (Corrected Model) ===')
    conf_model.load_state_dict(best_state_dict)
    conf_model.eval()

    # Mini-batch inference for corrected softmax on all nodes
    all_nodes_loader = NeighborLoader(
        data,
        input_nodes=torch.arange(data.num_nodes),
        num_neighbors=model_arguments.num_neighbors,
        batch_size=model_arguments.batch_size,
        shuffle=False,
        num_workers=4,
    )
    corrected_smx_all = torch.zeros(data.num_nodes, 2)
    with torch.no_grad():
        for batch in tqdm(all_nodes_loader, desc='Corrected inference'):
            batch = batch.to(device)
            n_seed = batch.batch_size
            delta = conf_model(batch.x, batch.edge_index)
            corrected = batch.x + delta
            corrected_smx = F.softmax(corrected[:n_seed], dim=1)
            corrected_smx_all[batch.n_id[:n_seed]] = corrected_smx.cpu()
    corrected_smx_final = corrected_smx_all.numpy()

    # Final accuracy
    test_idx_np = split_idx['test'].numpy()
    test_acc = (corrected_smx_all[test_idx_np].argmax(1).numpy() == labels_np[test_idx_np]).mean()
    base_test_acc = (base_smx_np[test_idx_np].argmax(1) == labels_np[test_idx_np]).mean()

    logging.info('Test accuracy — Base: %.4f, Corrected: %.4f', base_test_acc, test_acc)
    wandb.log({'test/base_acc': base_test_acc, 'test/corrected_acc': test_acc})

    for score in ['aps', 'raps']:
        # Base
        base_cov, base_eff, base_qhat = run_conformal_eval(
            base_smx_np, labels_np, val_idx, test_idx, alpha, score=score,
        )
        # Corrected
        corr_cov, corr_eff, corr_qhat = run_conformal_eval(
            corrected_smx_final, labels_np, val_idx, test_idx, alpha, score=score,
        )
        logging.info(
            '  %s Base:      coverage=%.4f efficiency=%.4f qhat=%.4f',
            score.upper(), base_cov, base_eff, base_qhat,
        )
        logging.info(
            '  %s Corrected: coverage=%.4f efficiency=%.4f qhat=%.4f',
            score.upper(), corr_cov, corr_eff, corr_qhat,
        )
        wandb.log({
            f'final/base_{score}_coverage': base_cov,
            f'final/base_{score}_efficiency': base_eff,
            f'final/corrected_{score}_coverage': corr_cov,
            f'final/corrected_{score}_efficiency': corr_eff,
        })

    # ---- ECE and NLL evaluation ----
    def _compute_ece(smx_np, labels, indices, n_bins=15):
        """Expected Calibration Error."""
        probs = smx_np[indices]
        true = labels[indices]
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == true).astype(float)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.sum() / len(indices) * abs(bin_acc - bin_conf)
        return ece

    def _compute_nll(smx_np, labels, indices):
        """Negative Log-Likelihood on test set."""
        probs = smx_np[indices]
        true = labels[indices]
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-7, 1.0)
        nll = -np.log(probs_clipped[np.arange(len(true)), true]).mean()
        return nll

    base_ece = _compute_ece(base_smx_np, labels_np, test_idx_np)
    corr_ece = _compute_ece(corrected_smx_final, labels_np, test_idx_np)
    base_nll = _compute_nll(base_smx_np, labels_np, test_idx_np)
    corr_nll = _compute_nll(corrected_smx_final, labels_np, test_idx_np)

    logging.info('=== Calibration Metrics (Test) ===')
    logging.info('  ECE  — Base: %.4f, Corrected: %.4f', base_ece, corr_ece)
    logging.info('  NLL  — Base: %.4f, Corrected: %.4f', base_nll, corr_nll)
    wandb.log({
        'test/base_ece': base_ece, 'test/corrected_ece': corr_ece,
        'test/base_nll': base_nll, 'test/corrected_nll': corr_nll,
    })

    logging.info('=== Classification UQ Complete ===')
