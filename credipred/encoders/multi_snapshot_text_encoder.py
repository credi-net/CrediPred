import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from credipred.encoders.encoder import Encoder
from credipred.utils.domain_handler import reverse_domain
from credipred.utils.readers import get_embeddings_multi_lookup


class MultiSnapTextEmbeddingEncoder(Encoder):
    def __init__(self, default_dimension: int, max_cache_size: int = 50):
        self.default_dimension = default_dimension
        self.max_cache_size = max_cache_size

    def __call__(
        self,
        domain_names: pd.Series,
        folder_locations: List[Path],
        lookup_names: List[str],
    ) -> Tensor:
        embeddings_lookup = get_embeddings_multi_lookup(
            [
                str(folder_location / lookup_name)
                for folder_location, lookup_name in zip(folder_locations, lookup_names)
            ]
        )
        text_embeddings_used = 0
        rni_used = 0
        n = len(domain_names)
        out = torch.empty(
            (n, self.default_dimension), dtype=torch.float32, device='cpu'
        )
        # Least Recently Used Cache (LRU)
        embedding_dict_cache: OrderedDict[str, Dict] = OrderedDict()

        for i, domain_name in tqdm(enumerate(domain_names), desc='Domain lookup'):
            rev_domain_name = reverse_domain(domain_name)

            if rev_domain_name in embeddings_lookup:
                file_info_list = embeddings_lookup[rev_domain_name]
                all_domain_embeddings = []

                for folder_path, wet_file_name in file_info_list:
                    cache_key = f'{folder_path}/{wet_file_name}'

                    if cache_key in embedding_dict_cache:
                        embedding_dict_cache.move_to_end(cache_key)
                    else:
                        if len(embedding_dict_cache) >= self.max_cache_size:
                            embedding_dict_cache.popitem(last=False)

                        full_path = folder_path / (wet_file_name + '.pkl')
                        if full_path.exists():
                            with open(full_path, 'rb') as file:
                                logging.info('Pushing to cache')
                                embedding_dict_cache[cache_key] = pickle.load(file)

                    file_data = embedding_dict_cache.get(cache_key, {})
                    if rev_domain_name in file_data:
                        entries = file_data[rev_domain_name]
                        all_domain_embeddings.extend(
                            [e[1] for e in entries if len(e) == 2]
                        )

                if all_domain_embeddings:
                    stacked_embs = torch.tensor(
                        np.array(all_domain_embeddings), dtype=torch.float32
                    )
                    aggregated_emb = torch.mean(stacked_embs, dim=0)
                    out[i] = aggregated_emb[0 : self.default_dimension]
                    text_embeddings_used += 1
                else:
                    out[i] = torch.rand(self.default_dimension)
                    rni_used += 1
            else:
                out[i] = torch.rand(self.default_dimension, dtype=torch.float32)
                rni_used += 1
        logging.info(f'Dimension of stacked embeddings: {out.shape}')
        logging.info(f'Text embeddings used: {text_embeddings_used}')
        logging.info(f'RNI used: {rni_used}')
        return out  # [num_domains, embedding_dim]
