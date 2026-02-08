import logging
import os
import pathlib
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

from credipred.encoders.encoder import Encoder
from credipred.utils.readers import (
    get_dqr_dict,
    get_full_dict,
    load_large_edge_csv_multi_snapshot,
    load_node_csv_multi_snapshot,
)
from credipred.utils.target_generation import (
    generate_exact_binary_targets_csv_multi_snapshots,
    generate_exact_targets_csv_multi_snapshot,
    reverse_domain,
)


class SquashedBinaryDatasetAllGlobalSplits(InMemoryDataset):
    def __init__(
        self,
        root: str,
        node_files: List[str] = ['features.csv'],
        edge_files: List[str] = ['edges.csv'],
        target_file: str = 'target.csv',
        target_col: str = 'weak_label',
        target_index_name: str = 'nid',
        target_index_col: int = 0,
        edge_src_col: str = 'src',
        edge_dst_col: str = 'dst',
        index_col: int = 1,
        index_name: str = 'node_id',
        force_undirected: bool = False,
        switch_source: bool = False,
        encoding: Optional[Dict[str, Encoder]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 42,
        processed_dir: Optional[str] = None,
        split_dir: Optional[str] = None,
        embedding_location: Optional[List[str]] = None,
        embedding_lookup: Optional[List[str]] = None,
    ):
        # We assume that the files are ordered decreasing recency (i.e most recent first)
        self.node_files = node_files
        self.edge_files = edge_files
        self.target_file = target_file
        self.target_col = target_col
        self.edge_src_col = edge_src_col
        self.edge_dst_col = edge_dst_col
        self.index_col = index_col
        self.index_name = index_name
        self.force_undirected = force_undirected
        self.switch_source = switch_source
        self.target_index_name = target_index_name
        self.target_index_col = target_index_col
        self.encoding = encoding
        self.seed = seed
        self.embedding_location: List[pathlib.Path] = []
        self.embedding_lookup = []
        if embedding_location:
            for el in embedding_location:
                self.embedding_location.append(pathlib.Path(el))
        else:
            self.embedding_location.append(pathlib.Path())

        logging.info(f'Embedding locations: {self.embedding_location}')
        if embedding_lookup:
            for el in embedding_lookup:
                self.embedding_lookup.append(el)
        logging.info(f'Embedding lookups: {self.embedding_lookup}')
        if split_dir:
            self.split_dir = pathlib.Path(split_dir)
            logging.info(f'Global split directory: {split_dir}')
        else:
            self.split_dir = pathlib.Path()

        self._custome_processed_dir = processed_dir
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        """Return the directory containing raw dataset files."""
        return os.path.join(self.root)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of expected raw file names."""
        return [self.node_file, self.edge_file]

    @property
    def processed_dir(self) -> str:
        """Return the directory used to store processed dataset files."""
        if self._custome_processed_dir is not None:
            return self._custome_processed_dir
        return super().processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return ['data.pt']

    def download(self) -> None:
        """No-op download hook (raw data must already exist locally)."""

    def process(self) -> None:
        """Generate targets, construct graph tensors, and create train/valid/test splits."""
        node_paths = []
        edge_paths = []
        for node_path in self.node_files:
            node_paths.append(os.path.join(self.raw_dir, node_path))

        for edge_path in self.edge_files:
            edge_paths.append(os.path.join(self.raw_dir, edge_path))

        target_path = os.path.join(self.raw_dir, self.target_file)
        if os.path.exists(target_path):
            logging.info('Target file already exists.')
        else:
            logging.info('Generating binary target file.')
            label_dict = get_full_dict()
            generate_exact_binary_targets_csv_multi_snapshots(
                node_paths, target_path, label_dict
            )

        logging.info('***Constructing Feature Matrix***')
        x_full, mapping, full_index = load_node_csv_multi_snapshot(
            paths=node_paths,
            embedding_location=self.embedding_location,
            embedding_lookup=self.embedding_lookup,
            index_col=0,
            encoders=self.encoding,
        )
        logging.info('***Feature Matrix Done***')

        if x_full is None:
            raise TypeError('X is None type. Please use an encoding.')

        df_target = pd.read_csv(target_path)
        logging.info(f'Size of target dataframe: {df_target.shape}')

        # mapping_index = [mapping[domain.strip()] for domain in df_target['domain']]
        mapping_index = []
        valid_indices = []
        for idx, row in df_target.iterrows():
            domain = row['domain'].strip()
            found = False

            if domain in mapping:
                mapping_index.append(mapping[domain])
                found = True
            else:
                rev_domain = reverse_domain(domain)
                if rev_domain in mapping:
                    mapping_index.append(mapping[rev_domain])
                    found = True
            if found:
                valid_indices.append(idx)
            else:
                logging.info(
                    f'Critical: {domain} and its reverse not found in mapping.'
                )

        df_target = df_target.loc[valid_indices].copy()

        df_target.index = mapping_index
        logging.info(f'Size of mapped target dataframe: {df_target.shape}')

        missing_idx = full_index.difference(mapping_index)
        filler = pd.DataFrame(
            {col: np.nan for col in df_target.columns}, index=missing_idx
        )
        df_target = pd.concat([df_target, filler])
        df_target.sort_index(inplace=True)
        logging.info(f'Size of filled target dataframe: {df_target.shape}')

        target_series = pd.to_numeric(df_target[self.target_col], errors='coerce')
        score_values = target_series.fillna(-1).astype('int32').values
        score = torch.tensor(
            score_values,
            dtype=torch.long,
        )
        logging.info(f'Size of score vector: {score.size()}')

        labeled_mask = score != -1.0
        labeled_idx = torch.nonzero(torch.tensor(labeled_mask), as_tuple=True)[0]
        labeled_scores = score[labeled_idx].squeeze().numpy()

        if labeled_scores.size == 0:
            raise ValueError(
                f"No labeled nodes found in target column '{self.target_col}'"
            )

        def get_split_indices(parquet_file: str, target_df: pd.DataFrame) -> List:
            df_global = pd.read_parquet(parquet_file)

            local_split_df = target_df[target_df['domain'].isin(df_global['domain'])]

            split_scores = pd.to_numeric(
                local_split_df[self.target_col], errors='coerce'
            ).fillna(-1)

            lost_domains = local_split_df[
                pd.to_numeric(local_split_df[self.target_col], errors='coerce').isna()
            ]

            logging.info(f'Domains lost due to invalid labels: {len(lost_domains)}')

            valid_mask = split_scores != -1.0
            indices = [
                mapping[d]
                for d in local_split_df.loc[valid_mask, 'domain']
                if d in mapping
            ]
            return indices

        train_idx_list = get_split_indices(
            parquet_file=str(self.split_dir / 'train_domains.parquet'),
            target_df=df_target,
        )
        logging.info(f'Size train: {len(train_idx_list)}')
        valid_idx_list = get_split_indices(
            parquet_file=str(self.split_dir / 'val_domains.parquet'),
            target_df=df_target,
        )
        logging.info(f'Size valid: {len(valid_idx_list)}')
        test_idx_list = get_split_indices(
            parquet_file=str(self.split_dir / 'test_domains.parquet'),
            target_df=df_target,
        )
        logging.info(f'Size test: {len(test_idx_list)}')

        train_idx = torch.as_tensor(train_idx_list)
        valid_idx = torch.as_tensor(valid_idx_list)
        test_idx = torch.as_tensor(test_idx_list)
        logging.info(f'Train size: {train_idx.size()}')

        idx_dict = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }

        logging.info('***Constructing Edge Matrix***')
        edge_index, edge_attr = load_large_edge_csv_multi_snapshot(
            paths=edge_paths,
            src_index_col=self.edge_src_col,
            dst_index_col=self.edge_dst_col,
            switch_source=self.switch_source,
            mapping=mapping,
            encoders=None,
        )
        logging.info('***Edge Matrix Constructed***')

        if self.force_undirected:
            logging.info('Converting edge index to undirected.')
            edge_index = to_undirected(edge_index)

        data = Data(x=x_full, y=score, edge_index=edge_index, edge_attr=edge_attr)
        data.labeled_mask = labeled_mask.detach().clone().bool()
        # Set global indices for our transductive nodes:
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.valid_mask[valid_idx] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        data.idx_dict = idx_dict

        self.verify_stratification(data=data, dict_split=data.idx_dict)

        assert data.edge_index.max() < data.x.size(0), 'edge_index out of bounds'

        torch.save(mapping, self.processed_dir + '/mapping.pt')
        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        """Return the stored train/valid/test index split.

        Returns:
            Dict
                Mapping with keys {'train', 'valid', 'test'} and index tensors.

        Raises:
            TypeError
                If the split is not available.
        """
        data = self[0]
        if hasattr(data, 'idx_dict') and data.idx_dict is not None:
            return data.idx_dict
        raise TypeError('idx split is empty.')

    def get_mapping(self) -> Dict:
        """Return the node ID mapping (lazy) from raw identifiers to internal indices."""
        if not hasattr(self, '_mapping'):
            self._mapping = torch.load(self.processed_dir + '/mapping.pt')
        return self._mapping

    def verify_stratification(
        self, data: Data, dict_split: Dict[str, torch.Tensor]
    ) -> None:
        logging.info('Label Balance Verification')
        for name, indices in dict_split.items():
            split_labels = data.y[indices]

            total = split_labels.size(0)
            pos_count = (split_labels == 1).sum().item()
            neg_count = (split_labels == 0).sum().item()

            pos_ratio = pos_count / total if total > 0 else 0

            logging.info(
                f'{name:5}: Total = {total:4} | Pos={pos_count:3} | Neg={neg_count:3} | Ratio={pos_ratio:.2%}'
            )


class SquashedDatasetGlobalSplit(InMemoryDataset):
    def __init__(
        self,
        root: str,
        node_files: List[str] = ['features.csv'],
        edge_files: List[str] = ['edges.csv'],
        target_file: str = 'target.csv',
        target_col: str = 'score',
        target_index_name: str = 'nid',
        target_index_col: int = 0,
        edge_src_col: str = 'src',
        edge_dst_col: str = 'dst',
        index_col: int = 1,
        index_name: str = 'node_id',
        force_undirected: bool = False,
        switch_source: bool = False,
        encoding: Optional[Dict[str, Encoder]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        seed: int = 42,
        processed_dir: Optional[str] = None,
        split_dir: Optional[str] = None,
        embedding_location: Optional[List[str]] = None,
        embedding_lookup: Optional[List[str]] = None,
    ):
        self.node_files = node_files
        self.edge_files = edge_files
        self.target_file = target_file
        self.target_col = target_col
        self.edge_src_col = edge_src_col
        self.edge_dst_col = edge_dst_col
        self.index_col = index_col
        self.index_name = index_name
        self.force_undirected = force_undirected
        self.switch_source = switch_source
        self.target_index_name = target_index_name
        self.target_index_col = target_index_col
        self.encoding = encoding
        self.seed = seed
        self._custome_processed_dir = processed_dir
        self.embedding_location: List[pathlib.Path] = []
        self.embedding_lookup = []

        if embedding_location:
            for el in embedding_location:
                self.embedding_location.append(pathlib.Path(el))
        else:
            self.embedding_location.append(pathlib.Path())

        logging.info(f'Embedding Locations: {self.embedding_location}')
        if embedding_lookup:
            for el in embedding_lookup:
                self.embedding_lookup.append(el)

        logging.info(f'Embedding lookups: {self.embedding_lookup}')
        logging.info(f'Split Directory: {split_dir}')
        if split_dir:
            self.split_dir = pathlib.Path(split_dir)
            logging.info(f'Global split directory: {split_dir}')
        else:
            self.split_dir = pathlib.Path()

        logging.info(f'{self.split_dir}')
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_dir(self) -> str:
        """Return the directory containing raw dataset files."""
        return os.path.join(self.root)

    @property
    def raw_file_names(self) -> List[str]:
        """Return the list of expected raw file names."""
        return [self.node_file, self.edge_file]

    @property
    def processed_dir(self) -> str:
        """Return the directory used to store processed dataset files."""
        if self._custome_processed_dir is not None:
            return self._custome_processed_dir
        return super().processed_dir

    @property
    def processed_file_names(self) -> List[str]:
        """Return the list of processed file names."""
        return ['data.pt']

    def download(self) -> None:
        """No-op download hook (raw data must already exist locally)."""

    def process(self) -> None:
        """Generate targets, construct graph tensors, and create train/valid/test splits."""
        node_paths = []
        edge_paths = []
        for node_path in self.node_files:
            node_paths.append(os.path.join(self.raw_dir, node_path))
        for edge_path in self.edge_files:
            edge_paths.append(os.path.join(self.raw_dir, edge_path))
        target_path = os.path.join(self.raw_dir, self.target_file)
        if os.path.exists(target_path):
            logging.info('Target file already exists.')
        else:
            logging.info('Generating target file.')
            dqr_dict = get_dqr_dict()
            generate_exact_targets_csv_multi_snapshot(node_paths, target_path, dqr_dict)

        logging.info(f'SPLIT DIR: {self.split_dir}')
        logging.info('***Constructing Feature Matrix***')
        x_full, mapping, full_index = load_node_csv_multi_snapshot(
            paths=node_paths,
            embedding_location=self.embedding_location,
            embedding_lookup=self.embedding_lookup,
            index_col=0,
            encoders=self.encoding,
        )
        logging.info('***Feature Matrix Done***')

        if x_full is None:
            raise TypeError('X is None type. Please use an encoding.')

        df_target = pd.read_csv(target_path)
        logging.info(f'Size of target dataframe: {df_target.shape}')

        mapping_index = [mapping[domain.strip()] for domain in df_target['domain']]
        df_target.index = mapping_index
        logging.info(f'Size of mapped target dataframe: {df_target.shape}')

        missing_idx = full_index.difference(mapping_index)
        filler = pd.DataFrame(
            {col: np.nan for col in df_target.columns}, index=missing_idx
        )
        df_target = pd.concat([df_target, filler])
        df_target.sort_index(inplace=True)
        logging.info(f'Size of filled target dataframe: {df_target.shape}')
        score = torch.tensor(
            df_target[self.target_col].astype('float32').fillna(-1).values,
            dtype=torch.float,
        )
        logging.info(f'Size of score vector: {score.size()}')

        labeled_mask = score != -1.0

        labeled_idx = torch.nonzero(torch.tensor(labeled_mask), as_tuple=True)[0]
        labeled_scores = score[labeled_idx].squeeze().numpy()

        if labeled_scores.size == 0:
            raise ValueError(
                f"No labeled nodes found in target column '{self.target_col}'"
            )

        def get_split_indices(
            parquet_file: str, target_df: pd.DataFrame
        ) -> torch.Tensor:
            df_split_domains = pd.read_parquet(parquet_file)

            split_indices = [
                mapping[domain.strip()]
                for domain in df_split_domains['domain']
                if domain.strip() in mapping
            ]

            return torch.tensor(split_indices, dtype=torch.long)

        train_idx = get_split_indices(
            str(self.split_dir / 'train_regression_domains.parquet'),
            target_df=df_target,
        )
        logging.info(f'Train size: {train_idx.size()}')
        valid_idx = get_split_indices(
            str(self.split_dir / 'val_regression_domains.parquet'), target_df=df_target
        )
        logging.info(f'Valid size: {valid_idx.size()}')
        test_idx = get_split_indices(
            str(self.split_dir / 'test_regression_domains.parquet'), target_df=df_target
        )
        logging.info(f'Test size: {test_idx.size()}')

        logging.info('***Constructing Edge Matrix***')
        edge_index, edge_attr = load_large_edge_csv_multi_snapshot(
            paths=edge_paths,
            src_index_col=self.edge_src_col,
            dst_index_col=self.edge_dst_col,
            switch_source=self.switch_source,
            mapping=mapping,
            encoders=None,
        )
        logging.info('***Edge Matrix Constructed***')

        if self.force_undirected:
            logging.info('Converting edge index to undirected.')
            edge_index = to_undirected(edge_index)

        data = Data(x=x_full, y=score, edge_index=edge_index, edge_attr=edge_attr)

        data.labeled_mask = labeled_mask.detach().clone().bool()

        # Set global indices for our transductive nodes:
        num_nodes = data.num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.valid_mask[valid_idx] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[test_idx] = True
        data.idx_dict = {
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx,
        }

        assert data.edge_index.max() < data.x.size(0), 'edge_index out of bounds'

        self.verify_stratification(data=data)

        torch.save(mapping, self.processed_dir + '/mapping.pt')
        torch.save(self.collate([data]), self.processed_paths[0])

    def get_idx_split(self) -> Dict:
        """Return the stored train/valid/test index split.

        Returns:
            Dict
                Mapping with keys {'train', 'valid', 'test'} and index tensors.

        Raises:
            TypeError
                If the split is not available.
        """
        data = self[0]
        if hasattr(data, 'idx_dict') and data.idx_dict is not None:
            return data.idx_dict
        raise TypeError('idx split is empty.')

    def get_mapping(self) -> Dict:
        """Return the node ID mapping (lazy) from raw identifiers to internal indices."""
        if not hasattr(self, '_mapping'):
            self._mapping = torch.load(self.processed_dir + '/mapping.pt')
        return self._mapping

    def verify_stratification(self, data: Data) -> None:
        """Log the distribution of regression scores across train/valid/test splits."""
        splits = data.idx_dict
        bins = [0, 0.1, 0.2, 0.5, 0.8, 1.0]
        bin_labels = ['<0.1', '0.1-0.2', '0.2-0.5', '0.5-0.8', '0.8-1.0']

        logging.info('Score distribution verification')

        summary_data = []
        for name, indices in splits.items():
            split_scores = data.y[indices].numpy()
            total_nodes = split_scores.size

            counts = np.histogram(split_scores, bins=bins)[0]

            precentages = (counts / total_nodes) * 100 if total_nodes > 0 else counts

            row = {'Split': name, 'Total': total_nodes}
            for label, pct in zip(bin_labels, precentages):
                row[label] = f'{pct:.1f}'

            summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        logging.info(f'\n{df_summary.to_string(index=False)}')
