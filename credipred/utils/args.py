import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import yaml
from hf_argparser import HfArgumentParser

from credipred.utils.path import get_root_dir, get_scratch


class Normalization(str, Enum):
    """Enumeration of supported normalization layers for the model."""

    NONE = 'none'
    LAYER_NORM = 'LayerNorm'
    BATCH_NORM = 'BatchNorm'


@dataclass
class MetaArguments:
    """Configuration for data locations, file paths, and global experiment settings."""

    log_file_path: Optional[str] = field(
        metadata={'help': 'Path to the log file to use.'},
    )
    node_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    edge_file: Union[str, List[str]] = field(
        metadata={
            'help': 'A csv or list of csv files containing the nodes of the graph.'
        },
    )
    target_file: Union[str, List[str]] = field(
        metadata={'help': 'A csv or list of csv files containing the targets.'},
    )
    database_folder: Union[str, List[str]] = field(
        metadata={'help': 'The folder containing the relational database.'},
    )
    processed_location: Union[str, List[str]] = field(
        metadata={'help': 'The location to save the processed feature matrix.'},
    )
    weights_directory: Union[str, List[str]] = field(
        metadata={'help': 'The location to save and load model weights.'},
    )
    target_col: str = field(
        default='cr_score',
        metadata={'help': 'The target column name in the target csv file.'},
    )
    edge_src_col: str = field(
        default='src', metadata={'help': 'The source column name in the edge file.'}
    )
    edge_dst_col: str = field(
        default='dst',
        metadata={'help': 'The destination column name in the edge file.'},
    )
    force_undirected: bool = field(
        default=False,
        metadata={'help': 'Forces the adjacency matrix to be undirected.'},
    )
    switch_source: bool = field(
        default=False,
        metadata={'help': 'Reverse src,dst order.'},
    )
    index_col: int = field(
        default=1,
        metadata={
            'help': 'The integer corresponding to the column denoting node ids in the feature csv file.'
        },
    )
    index_name: str = field(
        default='node_id',
        metadata={
            'help': 'The name of the index column. If index_col = 0, then this need not given.'
        },
    )
    encoder_dict: Dict[str, str] = field(
        default_factory=lambda: {
            'random': 'RNI',
            'pr_val': 'NORM',
            'hc_val': 'NORM',
            'text': 'TEXT',
            'pre': 'PRE',
        },
        metadata={
            'help': 'Node encoder dictionary defines which column is encoded by which encoder. Key: column, Value: Encoder'
        },
    )
    embedding_index_file: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the domain-to-embedding-file index pickle for PRE encoder.'},
    )
    embedding_folder: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to the folder containing embedding pickle files for PRE encoder.'},
    )
    global_seed: int = field(
        default=1337,
        metadata={'help': 'Random seed to use for reproducibiility.'},
    )
    is_scratch_location: bool = field(
        default=False,
        metadata={'help': 'Whether to use the /NOBACKUP/ or /SCRATCH/ disk on server.'},
    )

    def __post_init__(self) -> None:
        """Resolve all file and directory paths relative to the selected root directory."""
        # Select root directory
        root_dir = get_scratch() if self.is_scratch_location else get_root_dir()

        def resolve_paths(files: Union[str, List[str]]) -> Union[str, List[str]]:
            def resolve(f: str) -> str:
                # Force file to be relative to root_dir
                return str(root_dir / f.lstrip('/'))

            if isinstance(files, str):
                return resolve(files)
            return [resolve(f) for f in files]

        self.node_file = resolve_paths(self.node_file)
        self.edge_file = resolve_paths(self.edge_file)
        self.target_file = resolve_paths(self.target_file)
        self.database_folder = resolve_paths(self.database_folder)
        self.processed_location = resolve_paths(self.processed_location)

        # Resolve embedding paths if provided
        if self.embedding_index_file is not None:
            self.embedding_index_file = resolve_paths(self.embedding_index_file)
        if self.embedding_folder is not None:
            self.embedding_folder = resolve_paths(self.embedding_folder)

        if self.log_file_path is not None:
            self.log_file_path = str(get_root_dir() / self.log_file_path)


@dataclass
class DataArguments:
    """Configuration of task-level data and problem settings."""

    task_name: str = field(
        metadata={'help': 'The name of the task to train on'},
    )
    initial_encoding_col: str = field(
        default='random', metadata={'help': 'The initial input to the GNN.'}
    )
    num_test_shards: int = field(
        metadata={'help': 'Number of test splits to do for uncertainty estimates.'},
        default=1,
    )
    is_regression: bool = field(
        default=False,
        metadata={'help': 'Is the task a regression or classification problem'},
    )


@dataclass
class ModelArguments:
    """Configuration of model architecture and training hyperparameters."""

    model: str = field(
        default='GCN',
        metadata={'help': 'Model identifer for the GNN.'},
    )
    num_layers: int = field(
        default=3,
        metadata={'help': 'Number of layers in GNN or iterations in message passing.'},
    )
    hidden_channels: int = field(
        default=256, metadata={'help': 'Inner dimension of update weight matrix.'}
    )
    normalization: str = field(
        default=Normalization.BATCH_NORM,
        metadata={
            'help': 'The normalization method. Choices: none, LayerNorm or BatchNorm.'
        },
    )
    num_neighbors: list[int] = field(
        default_factory=lambda: [
            -1
        ],  # TODO: Where do MEM errors occur, what is the size?
        metadata={'help': 'Number of neighbors in Neighbor Loader.'},
    )
    batch_size: int = field(
        default=128, metadata={'help': 'Batch size in Neighbor loader.'}
    )
    embedding_dimension: int = field(
        default=128, metadata={'help': 'The output dimension of the GNN.'}
    )
    dropout: float = field(default=0.1, metadata={'help': 'Dropout value.'})
    lr: float = field(default=0.001, metadata={'help': 'Learning Rate.'})
    weight_decay: float = field(
        default=0.0, metadata={'help': 'Weight decay (L2 regularization) for optimizer.'}
    )
    epochs: int = field(default=500, metadata={'help': 'Number of epochs.'})
    patience: int = field(
        default=0,
        metadata={
            'help': 'Early stopping patience. 0 means no early stopping. '
            'Training stops if validation loss does not improve for this many epochs.'
        },
    )
    runs: int = field(default=100, metadata={'help': 'Number of trials.'})
    use_cuda: bool = field(default=True, metadata={'help': 'Whether to use cuda.'})
    device: int = field(default=0, metadata={'help': 'Device to be used.'})
    log_steps: int = field(
        default=50, metadata={'help': 'Step mod epoch to print logger.'}
    )
    # GraphGPS specific parameters
    gps_heads: int = field(
        default=4,
        metadata={'help': 'Number of attention heads for GraphGPS global attention.'},
    )
    gps_attn_type: str = field(
        default='multihead',
        metadata={
            'help': "Attention type for GraphGPS. Choices: 'multihead' or 'performer'."
        },
    )
    gps_attn_dropout: float = field(
        default=0.1,
        metadata={'help': 'Dropout rate for GraphGPS attention.'},
    )
    gps_local_mpnn: str = field(
        default='gin',
        metadata={
            'help': "Local MPNN type for GraphGPS. Choices: 'gin', 'gatedgcn', 'gat'."
        },
    )


@dataclass
class ExperimentArgument:
    """Container for a single experiment's data and model configuration."""

    data_args: DataArguments = field(
        metadata={'help': 'Data arguments for GNN configuration.'}
    )
    model_args: ModelArguments = field(
        metadata={'help': 'Model arguments for the GNN.'}
    )


@dataclass
class ExperimentArguments:
    """Collection of named experiments and their configurations."""

    exp_args: Dict[str, ExperimentArgument] = field(
        metadata={'help': 'List of experiments.'}
    )

    def __post_init__(self) -> None:
        """Convert experiment dictionaries into ExperimentArgument instances."""

        def _remap_experiment_args(
            experiments: Dict[str, ExperimentArgument],
        ) -> Dict[str, ExperimentArgument]:
            for exp_name, exp_val in experiments.items():
                if isinstance(exp_val, dict):
                    model_args = ModelArguments(**exp_val['model_args'])
                    data_args = DataArguments(**exp_val['data_args'])
                    experiments[exp_name] = ExperimentArgument(
                        model_args=model_args,
                        data_args=data_args,
                    )
            return experiments

        self.exp_args = _remap_experiment_args(self.exp_args)


def parse_args(
    config_yaml: Union[str, pathlib.Path],
) -> Tuple[MetaArguments, ExperimentArguments]:
    """Parse a YAML configuration file into typed argument objects.

    Parameters:
        config_yaml : Union[str, pathlib.Path]
            Path to the YAML configuration file.

    Returns:
        Tuple[MetaArguments, ExperimentArguments]
            Parsed meta and experiment configuration objects.
    """
    config_dict = yaml.safe_load(pathlib.Path(config_yaml).read_text())
    config_dict = config_dict['MetaArguments'] | config_dict['ExperimentArguments']
    parser = HfArgumentParser((MetaArguments, ExperimentArguments))
    return parser.parse_dict(config_dict, allow_extra_keys=True)
