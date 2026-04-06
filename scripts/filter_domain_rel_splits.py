import argparse
import logging

import numpy as np
import pandas as pd

from credipred.utils.logger import setup_logging
from credipred.utils.path import get_scratch

parser = argparse.ArgumentParser(
    description='Filter Domain Rel Splits.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--split-file',
    type=str,
    default='',
    help='Split file.',
)
parser.add_argument(
    '--output-dir',
    type=str,
    default='',
    help='Output directory.',
)
parser.add_argument(
    '--domains-annotated',
    type=str,
    default='',
    help='File containing annotated labels.',
)
parser.add_argument(
    '--category',
    type=str,
    default='phishing',
    help='File containing annotated labels.',
    choices=['general', 'phishing', 'misinfo', 'malware'],
)
parser.add_argument(
    '--convert_labels',
    action='store_true',
    help='File containing annotated labels.',
)


category_to_sub_category: dict[str, list] = {
    'general': [
        'wikipedia',
    ],
    'phishing': [
        'legit-phish',
        'phish-and-legit',
        'phish-dataset',
        'url-phish',
    ],
    'misinfo': [
        'misinfo-domains',
        'nelez',
    ],
    'malware': [
        'urlhaus',
    ],
}


def get_statistics(
    split_df: pd.DataFrame, labels_annotation_df: pd.DataFrame
) -> pd.DataFrame:
    stats = {'domains_occuring_in_labels': 0}
    annotated_domains = labels_annotation_df['domain'].unique()

    mask = split_df['domain'].isin(annotated_domains)

    count = split_df.loc[mask, 'domain'].nunique()

    stats['domains_occuring_in_labels'] = count

    return pd.DataFrame(stats)


def main() -> None:
    args = parser.parse_args()
    setup_logging('KDD_Filtering.log')
    root = get_scratch()
    split_file = root / args.split_file
    domains_annotated_file = root / args.domains_annotated
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    relevant_cols = category_to_sub_category[args.category]

    split_df = pd.read_parquet(split_file)
    number_of_original_domains = len(split_df)
    domains_annoated_df = pd.read_csv(domains_annotated_file)

    # Find domains where at least one of these columns is 1.
    is_in_category = domains_annoated_df[relevant_cols].any(axis=1)
    valid_domains = domains_annoated_df.loc[is_in_category, 'domain'].unique()

    logging.info(f'Stats: {get_statistics(split_df, domains_annoated_df).head()}')

    if args.convert_labels:
        valid_mask = split_df['domain'].isin(valid_domains)
        split_df['label'] = np.where(split_df[valid_mask], 1, 0)

    else:
        split_df = split_df[split_df['domain'].isin(valid_domains)]

    output_path = output_dir / f'filtered_{args.category}_{split_file.name}'
    split_df.to_parquet(output_path)
    logging.info(
        f'Filtered dataframe saved to {output_path}. Rows split: {number_of_original_domains}, Rows filtered: {len(split_df)}'
    )


if __name__ == '__main__':
    main()
