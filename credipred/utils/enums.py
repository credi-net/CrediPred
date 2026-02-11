from enum import Enum


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
