"""Layer A — Part 1: materialize per-domain conformal lower bounds + full mount-independence snapshot.

Pipeline (see plan repo-a-warm-beaver.md):
  Step 0  Snapshot domain->nid map, sorted calibration conformity scores, and
          val/test split node-ids + labels to STABLE storage. After this, qhat(alpha)
          for any alpha and gating of any retrieved domain are mount-independent.
  1-2     Compute qhat at alpha_node in {0.05, 0.1} from the validation split (CQR).
  3       Join domain_ratings.csv (forward domains) -> reversed/eTLD+1 -> nid -> lower_adj.
  4       Emit rated_domain_lower_bounds.csv + coverage/Spearman/histogram summary.

Golden checks: empirical coverage ~= 1-alpha on the test split (validates column order,
qhat sign, split correctness simultaneously); reproduce qhat ~= 0.2569 at the matching alpha.

CPU only, no keys, no network. Run inside the CrediPred uv venv:
  uv run python experiments/layer_a/materialize_lower_bounds.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from credipred.conformal_regression.cqr import (
    adjust_intervals,
    compute_cqr_scores,
    compute_qhat,
)

# ---- Paths --------------------------------------------------------------------
DG = Path('/mnt/data/slurm-storage/jiaxio/data/CrediBench/dec2024')  # data-gauss (flaky)
BASE_PREDS = Path(
    '/mnt/data/slurm-storage/jiaxio/ws/data/dec_2024/weights_quantile/pc1/GAT/base_preds.pt'
)  # stable
MAPPING = DG / 'processed_dir/pc1/mapping.pt'
RATINGS = Path('/mnt/data/slurm-storage/jiaxio/ws/context/data/domain_ratings.csv')  # stable
OUT = Path('/mnt/data/slurm-storage/jiaxio/ws/data/layer_a')  # stable
SNAP = OUT / 'snapshot'

ALPHAS = (0.05, 0.1)

# Column convention of base_preds.pt: [mid, lower, upper]  (verified: col2-col1==logged width)
MID, LOWER, UPPER = 0, 1, 2


# ---- Domain normalization (graph keys are reversed, TLD-first) ---------------
def reverse_domain(domain: str) -> str:
    return '.'.join(reversed(domain.strip().lower().strip('.').split('.')))


def domain_variants_reversed(domain: str) -> list[str]:
    """Reversed-form lookup keys for a forward domain. Try as-is and www-stripped."""
    d = domain.strip().lower().strip('.')
    if d.startswith('www.'):
        d = d[4:]
    out = [reverse_domain(d)]
    # also try eTLD+1 fold (last 2 labels) in case the rated domain carries a subdomain
    parts = d.split('.')
    if len(parts) > 2:
        out.append(reverse_domain('.'.join(parts[-2:])))
    return out


def lookup_nid(domain: str, mapping: dict) -> int | None:
    for key in domain_variants_reversed(domain):
        nid = mapping.get(key)
        if nid is not None:
            return int(nid)
    return None


# ---- Load splits (parquet carries domain + label(pc1)) -----------------------
def load_split(name: str, mapping: dict):
    df = pd.read_parquet(DG / f'{name}.parquet')  # cols: domain (reversed), label
    nids, ys = [], []
    for dom, lab in zip(df['domain'], df['label']):
        nid = mapping.get(str(dom).strip())  # split domains are already reversed
        if nid is not None:
            nids.append(int(nid))
            ys.append(float(lab))
    return torch.tensor(nids, dtype=torch.long), torch.tensor(ys, dtype=torch.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SNAP.mkdir(parents=True, exist_ok=True)

    print('=== Loading inputs ===', flush=True)
    mapping = torch.load(MAPPING, weights_only=False)
    base_preds = torch.load(BASE_PREDS, map_location='cpu', weights_only=True)
    print(f'mapping: {len(mapping):,} domains | base_preds: {tuple(base_preds.shape)}', flush=True)

    val_nid, val_y = load_split('val_regression_domains', mapping)
    test_nid, test_y = load_split('test_regression_domains', mapping)
    print(f'calib(val)={len(val_nid)} test={len(test_nid)}', flush=True)

    lower_val = base_preds[val_nid, LOWER]
    upper_val = base_preds[val_nid, UPPER]
    cal_scores = compute_cqr_scores(lower_val, upper_val, val_y)
    cal_scores_sorted, _ = torch.sort(cal_scores)

    # ---- Step 0: snapshot mount-independent artifacts to STABLE storage -------
    print('\n=== Step 0: snapshot to stable storage ===', flush=True)
    shutil.copy2(MAPPING, SNAP / 'mapping_pc1.pt')
    torch.save(cal_scores_sorted, SNAP / 'cal_scores_sorted.pt')
    torch.save({'val_nid': val_nid, 'val_y': val_y, 'test_nid': test_nid, 'test_y': test_y},
               SNAP / 'splits.pt')
    print(f'  snapshot -> {SNAP} (mapping_pc1.pt, cal_scores_sorted.pt, splits.pt)', flush=True)
    print('  qhat(alpha) is now a pure quantile lookup on cal_scores_sorted (offline).', flush=True)

    # ---- 1-2: qhat at each alpha + GOLDEN CHECK (empirical coverage ~ 1-alpha) -
    print('\n=== qhat + golden coverage check ===', flush=True)
    qhats = {}
    lower_test = base_preds[test_nid, LOWER]
    upper_test = base_preds[test_nid, UPPER]
    for alpha in ALPHAS:
        qhat = compute_qhat(cal_scores, alpha)
        qhats[alpha] = qhat
        lo_adj, hi_adj = adjust_intervals(lower_test, upper_test, qhat)
        cov = ((test_y >= lo_adj) & (test_y <= hi_adj)).float().mean().item()
        width = (hi_adj - lo_adj).mean().item()
        print(f'  alpha={alpha}: qhat={qhat:.4f}  test coverage={cov:.4f} (target {1 - alpha:.2f})  '
              f'width={width:.4f}', flush=True)

    # ---- 3-4: materialize rated-domain lower bounds ---------------------------
    print('\n=== Materialize rated-domain lower bounds ===', flush=True)
    ratings = pd.read_csv(RATINGS)
    rows = []
    n_in_graph = 0
    for dom, pc1 in zip(ratings['domain'], ratings['pc1']):
        nid = lookup_nid(str(dom), mapping)
        in_graph = nid is not None
        n_in_graph += int(in_graph)
        rec = {'domain': dom, 'pc1': pc1, 'in_graph': in_graph, 'nid': nid if in_graph else -1}
        if in_graph:
            mid = base_preds[nid, MID].item()
            lo = base_preds[nid, LOWER].item()
            hi = base_preds[nid, UPPER].item()
            rec['mid'] = mid
            rec['lower'] = lo
            rec['upper'] = hi
            for alpha in ALPHAS:
                q = qhats[alpha]
                rec[f'lower_adj_a{alpha}'] = lo - q
                rec[f'upper_adj_a{alpha}'] = hi + q
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_csv = OUT / 'rated_domain_lower_bounds.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'  wrote {out_csv}  ({len(out_df)} rows, {n_in_graph} in-graph)', flush=True)

    # ---- Summary: coverage stat, Spearman, lower_adj histogram ----------------
    g = out_df[out_df['in_graph']].copy()
    print('\n=== Coverage / quality summary ===', flush=True)
    print(f'  rated_in_graph / total = {n_in_graph} / {len(out_df)} = {n_in_graph / len(out_df):.3f}',
          flush=True)
    rho, p = spearmanr(g['mid'], g['pc1'])
    print(f'  Spearman rho(mid vs pc1) = {rho:.4f} (p={p:.2e}, n={len(g)})', flush=True)
    # quantile-crossing note: mid can exceed upper (mid head not monotone-constrained)
    n_cross = int((g['mid'] > g['upper']).sum())
    print(f'  quantile-crossing (mid>upper): {n_cross}/{len(g)} rated-in-graph rows', flush=True)
    la = g[f'lower_adj_a{ALPHAS[0]}'].to_numpy()
    hist, edges = np.histogram(la, bins=10)
    print(f'  lower_adj (alpha={ALPHAS[0]}) histogram:', flush=True)
    for c, lo_e, hi_e in zip(hist, edges[:-1], edges[1:]):
        print(f'    [{lo_e:+.3f},{hi_e:+.3f}): {c}', flush=True)
    n_floor = int((la <= 0).sum())
    print(f'  lower_adj <= 0 (floored): {n_floor}/{len(g)} '
          f'({100 * n_floor / len(g):.1f}%) -> dictates lambda-sweep granularity', flush=True)

    print('\n=== DONE ===', flush=True)


if __name__ == '__main__':
    main()
