from enum import Enum, auto


class Scoring(str, Enum):
    mse = 'MSE'
    r2 = 'R2'
    mae = 'MAE'
    acc = 'Accuracy'


class Metric(str, Enum):
    acc = 'Accuracy'
    loss = 'Loss'


class Label(str, Enum):
    pc1 = 'PC1'
    mbfc = 'MBFC-BIAS'


class ConvolutionEnums(str, Enum):
    gcn = 'GCN'
    sage = 'SAGE'
    gat = 'GAT'
    gatv2 = 'GATv2'


class TrainingMethods(Enum):
    DEFAULT = auto()
    DOWN_SAMPLE = auto()
    WEIGHTED_LOSS = auto()
