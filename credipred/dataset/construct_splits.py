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

parser = argparse.ArgumentParser(
    description='Split on all months.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


def generate_splits(
    all_domains: List[str],
    domains_to_binary: Dict[str, int],
    scratch_path: pathlib.Path,
) -> None:
    labeled_data = []
    for domain in tqdm(all_domains, desc='Intersecting with Labels'):
        try:
            if domain in domains_to_binary:
                domain_to_label = {'domain': domain, 'label': domains_to_binary[domain]}
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

    test_df, val_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    output_dir = scratch_path / 'data' / 'splits'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / 'train_domains.parquet', index=False)
    test_df.to_parquet(output_dir / 'test_domains.parquet', index=False)
    val_df.to_parquet(output_dir / 'val_domains.parquet', index=False)

    logging.info(f'Splits saved to {output_dir}')
    logging.info(
        f'Counts - Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}'
    )


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
