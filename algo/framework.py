import tensorflow as tf
import dataclasses
import numpy as np
import logging
from tqdm import tqdm
from abc import ABC


@dataclasses.dataclass
class AlgoConfig(ABC):
    """This defines the common parameters for the algorithm"""
    learning_rate: float = 0.0001
    training_epochs: int = 50
    distributed_training: bool = False
    pretrained_model_path: str = ''
    evaluation_period: int = 5


@dataclasses.dataclass
class Algo(ABC):
    def __init__(self) -> None:
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        pass

    def optimize(self, train_minibatch) -> np.ndarray:
        pass

    def _optimize(self, inputs, labels):
        pass

    def _validate(self, inputs, labels):
        pass

    def validate(self, val_minibatch) -> np.ndarray:
        pass

    def predict(self, inputs):
        pass

    def save_model(self, path):
        pass


def val(val_data, model, writer):
    pass


def distributed_train_step():
    pass


def distributed_val_step():
    pass








