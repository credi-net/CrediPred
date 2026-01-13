import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from credipred.encoders.encoder import Encoder
from credipred.utils.domain_handler import reverse_domain


class TextEmbeddingEncoder(Encoder):
    def __init__(self, default_dimension: int, max_cache_size: int = 4):
        self.default_dimension = default_dimension
        self.max_cache_size = max_cache_size

    def __call__(
        self,
        domain_names: pd.Series,
        embeddings_lookup: Dict[str, str],
        folder_location: Path,
    ) -> Tensor:
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
                wet_file_name = embeddings_lookup[rev_domain_name]
                if wet_file_name in embedding_dict_cache:
                    embedding_dict_cache.move_to_end(wet_file_name)
                else:
                    if len(embedding_dict_cache) >= self.max_cache_size:
                        embedding_dict_cache.popitem(last=False)

                    path = folder_location / (wet_file_name + '.pkl')
                    with open(path, 'rb') as file:
                        embedding_dict_cache[wet_file_name] = pickle.load(file)

                entries = embedding_dict_cache[wet_file_name][rev_domain_name]
                embeddings = [e[1] for e in entries if len(e) == 2]
                stacked_embs = torch.tensor(np.array(embeddings), dtype=torch.float32)
                aggregated_emb = torch.mean(stacked_embs, dim=0)
                out[i] = aggregated_emb[0 : self.default_dimension]
                text_embeddings_used += 1
            else:
                out[i] = torch.rand(self.default_dimension, dtype=torch.float32)
                rni_used += 1
        logging.info(f'Dimension of stacked embeddings: {out.shape}')
        logging.info(f'Text embeddings used: {text_embeddings_used}')
        logging.info(f'RNI used: {rni_used}')
        return out  # [num_domains, embedding_dim]
