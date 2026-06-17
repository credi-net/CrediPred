"""Layer A — Phase 0: retrieved-domain three-bucket hit-rate + closed-world lower-bound table.

The DECISION number: of the domains actually retrieved by the RARR pipeline, how many are
  (1) rated      -> in domain_ratings.csv (A1's third-party signal applies)
  (2) in-graph    -> a CrediBench node but unrated (GNN gives an interval -> the GNN's value prop)
  (3) unseen      -> not in the graph at all (gating policy undefined; size gates eval validity)

Retrieved-domain universe = the URLs in the passage cache (frozen at Phase 0 end; the RARR
evidence 3-tuple drops the URL, but the cache keeps it). Closed-world deliverable: a second
materialization over this frozen set -> retrieved_domain_lower_bounds.csv (Phase 1 = pure lookup).

Domain matching (M3): exact host first, then eTLD+1 fallback BEFORE declaring unseen, so a
host vs registered-domain mismatch does not inflate bucket-3 and pollute the decision number.

Mount-independent: reads only the snapshot (mapping_pc1.pt, cal_scores_sorted.pt) + base_preds.pt,
all on stable storage. CPU only, no keys.

  uv run python experiments/layer_a/three_bucket_hitrate.py [--cache-dir <.rarr_cache>] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import torch

WS = Path('/mnt/data/slurm-storage/jiaxio/ws')
SNAP = WS / 'data/layer_a/snapshot'
BASE_PREDS = WS / 'data/dec_2024/weights_quantile/pc1/GAT/base_preds.pt'
RATED_CSV = WS / 'data/layer_a/rated_domain_lower_bounds.csv'
OUT = WS / 'data/layer_a'
DEFAULT_CACHE = WS / 'context/.rarr_cache/passages'
MID, LOWER, UPPER = 0, 1, 2

# Mirrors context search.py _registrable_domain so bucketing matches the pipeline's normalization.
_SECOND_LEVEL_CC = {'co', 'com', 'org', 'net', 'gov', 'edu', 'ac'}
_CC_TLDS = {'uk', 'au', 'jp', 'nz', 'za'}


def registrable_domain(host: str) -> str:
    host = (host or '').strip().lower().strip('.')
    if host.startswith('www.'):
        host = host[4:]
    parts = [p for p in host.split('.') if p]
    if len(parts) <= 2:
        return host
    if len(parts) >= 3 and parts[-1] in _CC_TLDS and parts[-2] in _SECOND_LEVEL_CC:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def extract_host(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or '').split(':')[0].strip().lower()
    if not host:
        host = (parsed.path or '').split('/')[0].split(':')[0].strip().lower()
    if host.startswith('www.'):
        host = host[4:]
    return host


def reverse_domain(domain: str) -> str:
    return '.'.join(reversed(domain.strip().lower().strip('.').split('.')))


def qhat_from_snapshot(alpha: float) -> float:
    scores = torch.load(SNAP / 'cal_scores_sorted.pt', weights_only=True)
    n = scores.numel()
    q_level = min(max(math.ceil((n + 1) * (1 - alpha)) / n, 1e-6), 1.0)
    return torch.quantile(scores, q_level, interpolation='higher').item()


def collect_retrieved_urls(cache_dir: Path) -> list[str]:
    urls: list[str] = []
    for f in sorted(cache_dir.glob('*.json')):
        try:
            passages = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        for p in passages:
            u = p.get('url')
            if u:
                urls.append(u)
    return urls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', default=str(DEFAULT_CACHE))
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise SystemExit(f'passage cache not found: {cache_dir} (run Phase 0 first)')

    print('=== Loading snapshot + base_preds ===', flush=True)
    mapping = torch.load(SNAP / 'mapping_pc1.pt', weights_only=False)
    base_preds = torch.load(BASE_PREDS, map_location='cpu', weights_only=True)
    qhat = qhat_from_snapshot(args.alpha)
    rated = pd.read_csv(RATED_CSV)
    rated_set = set(rated['domain'].astype(str).str.lower())
    print(f'mapping={len(mapping):,} rated={len(rated_set)} qhat(a={args.alpha})={qhat:.4f}', flush=True)

    urls = collect_retrieved_urls(cache_dir)
    # registered-domain granularity for the decision number (one row per unique registered domain)
    reg_domains = sorted({registrable_domain(extract_host(u)) for u in urls if extract_host(u)})
    print(f'retrieved URLs={len(urls)}  unique registered domains={len(reg_domains)}', flush=True)

    def graph_nid(host: str):
        """Exact host first, then eTLD+1 fallback (M3). Returns (nid, granularity) or (None,'unseen')."""
        nid = mapping.get(reverse_domain(host))
        if nid is not None:
            return int(nid), 'host'
        reg = registrable_domain(host)
        if reg != host:
            nid = mapping.get(reverse_domain(reg))
            if nid is not None:
                return int(nid), 'etld1'
        return None, 'unseen'

    rows = []
    buckets = Counter()
    for dom in reg_domains:
        is_rated = dom in rated_set
        nid, gran = graph_nid(dom)
        in_graph = nid is not None
        if is_rated:
            bucket = 'rated'
        elif in_graph:
            bucket = 'in_graph_unrated'
        else:
            bucket = 'unseen'
        buckets[bucket] += 1
        rec = {'domain': dom, 'bucket': bucket, 'rated': is_rated,
               'in_graph': in_graph, 'graph_granularity': gran,
               'nid': nid if in_graph else -1}
        if in_graph:
            rec['mid'] = base_preds[nid, MID].item()
            rec['lower'] = base_preds[nid, LOWER].item()
            rec['lower_adj'] = base_preds[nid, LOWER].item() - qhat
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    out_csv = OUT / 'retrieved_domain_lower_bounds.csv'
    out_df.to_csv(out_csv, index=False)

    total = len(reg_domains)
    report = {
        'cache_dir': str(cache_dir),
        'alpha_node': args.alpha,
        'qhat': qhat,
        'retrieved_urls': len(urls),
        'unique_registered_domains': total,
        'buckets': dict(buckets),
        'bucket_fractions': {k: round(v / total, 4) for k, v in buckets.items()} if total else {},
        'gateable_fraction_in_graph': round((buckets['rated'] + buckets['in_graph_unrated']) / total, 4) if total else 0.0,
        'note': 'DATASET-CONDITIONAL (LIAR=PolitiFact political -> optimistic; re-measure on fringe datasets).',
    }
    (OUT / 'three_bucket_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\n=== THREE-BUCKET HIT-RATE (the decision number) ===', flush=True)
    for b in ('rated', 'in_graph_unrated', 'unseen'):
        c = buckets.get(b, 0)
        print(f'  {b:18s}: {c:5d}  ({100 * c / total:.1f}%)' if total else f'  {b}: 0', flush=True)
    print(f'  gateable (rated+in_graph) = {report["gateable_fraction_in_graph"] * 100:.1f}%', flush=True)
    print(f'\n  wrote {out_csv} + three_bucket_report.json', flush=True)
    print('  CAVEAT: dataset-conditional (LIAR optimistic); re-measure on rumors/fringe.', flush=True)


if __name__ == '__main__':
    main()
