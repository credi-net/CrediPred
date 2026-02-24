import argparse
import logging
import pathlib
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from credipred.utils.logger import setup_logging
from credipred.utils.path import get_scratch
from credipred.utils.readers import get_full_dict
from credipred.utils.seed import seed_everything
from credipred.utils.target_generation import strict_exact_etld1_match

parser = argparse.ArgumentParser(
    description='Split on all months.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


def downsample_to_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['label'].value_counts()
    minority_size = counts.min()
    balanced_df = df.groupby('label').sample(n=minority_size, random_state=42)
    return balanced_df.sample(frac=1, random_state=42)


def generate_splits(
    all_domains: List[str],
    domains_to_binary: Dict[str, int],
    scratch_path: pathlib.Path,
) -> None:
    labeled_data = []
    for domain in tqdm(all_domains, desc='Intersecting with Labels'):
        try:
            raw_domain = domain.strip()
            if not raw_domain:
                continue
            etld1 = strict_exact_etld1_match(raw_domain, domains_to_binary)
            if etld1 is None:
                continue
            metrics = domains_to_binary[etld1]
            domain_to_label = {'domain': raw_domain, 'label': metrics}
            labeled_data.append(domain_to_label)
        except KeyError:
            logging.info(f'Critical {domain.strip()} is not found in mapping.')

    if not labeled_data:
        logging.error('Labeled data is empty. No intersection found')
        return

    logging.info(f'Number of labeled domains: {len(labeled_data)}')

    df = pd.DataFrame(labeled_data)
    logging.info(f'Total unique labeled domains found: {len(df)}')

    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=42, stratify=df['label']
    )

    test_df_raw, val_df_raw = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    logging.info('Downsampling Test and Val sets to balance classess...')
    test_df = downsample_to_balance(test_df_raw)
    val_df = downsample_to_balance(val_df_raw)

    output_dir = scratch_path / 'data' / 'splits' / 'balanced'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / 'train_domains.parquet', index=False)
    test_df.to_parquet(output_dir / 'test_domains.parquet', index=False)
    val_df.to_parquet(output_dir / 'val_domains.parquet', index=False)

    logging.info(f'Splits saved to {output_dir}')
    logging.info(
        f'Counts - Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}'
    )

    logging.info(f'Train class Dist:\n{train_df["label"].value_counts()}')
    logging.info(f'Train class Dist:\n{test_df["label"].value_counts()}')
    logging.info(f'Train class Dist:\n{val_df["label"].value_counts()}')


def main() -> None:
    seed_everything(42)  # Ensuring reproducibility
    scratch = get_scratch()
    setup_logging('generate_all_targets.log')

    domains_to_binary = get_full_dict()
    paths = ['dec_2024_domain', 'nov2024_domain', 'oct2024_domain']

    all_unique_domains = set()

    for path_name in tqdm(paths, desc=f'Vertices'):
        node_csv = scratch / 'data' / path_name / 'vertices.csv'
        if node_csv.exists():
            logging.info(f'Reading {node_csv}...')
            # Reading only the 'domain' column to save memory
            df_chunk = pd.read_csv(node_csv, usecols=['domain'])
            logging.info('Loaded csv')
            logging.info(f'Logging Head: {df_chunk.head()}')
            all_unique_domains.update(df_chunk['domain'].tolist())
            logging.info(f'Number of unique domains {len(all_unique_domains)}')
            logging.info(f'Added csv: {node_csv} to set.')
        else:
            logging.warning(f'File not found: {node_csv}')

    if all_unique_domains:
        generate_splits(list(all_unique_domains), domains_to_binary, scratch)
    else:
        logging.error('No domains were loaded. Check your file paths.')


if __name__ == '__main__':
    main()
