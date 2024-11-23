import logging
import weakref
from copy import deepcopy
from typing import Optional

import lightning
import numpy as np
import torch

from kooplearn._src.serialization import pickle_load, pickle_save
from kooplearn.abc import ContextWindowDataset, TrainableFeatureMap

logger = logging.getLogger("kooplearn")


class NNFeatureMap(TrainableFeatureMap):
    """Implements a generic Neural Network feature maps. Can be used in conjunction to :class:`kooplearn.models.Nonlinear` to learn a Koopman/Transfer operator from data. The NN feature map is trained using the :class:`lightning.LightningModule` API, and can be trained using the :class:`lightning.Trainer` API. See the `PyTorch Lightning documentation <https://pytorch-lightning.readthedocs.io/en/latest/>`_ for more information.

    Args:
        encoder (torch.nn.Module): 
        loss_fn (torch.nn.Module): Loss function from :class:`kooplearn.nn`
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        loss_fn: torch.nn.Module,
    ):
        self.encoder = encoder
        self.loss_fn = loss_fn

        self._lookback_len = -1  # Dummy init value, will be determined at fit time.
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    def lookback_len(self):
        return self._lookback_len

    def save(self, filename):
        """Serialize the model to a file.

        Args:
            filename (path-like or file-like): Save the model to file.
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, filename):
        """Load a serialized model from a file.

        Args:
            filename (path-like or file-like): Load the model from file.

        Returns:
            The loaded model.
        """
        raise NotImplementedError()

    def fit(
        self,
        data,
    ):
        """
        """
        representations = self.encoder(data) # [deq_iterations, B, koopman_dim]
        loss = 0
        for t in range(len(representations) -1 ):
            loss += self.loss_fn(representations[t], representations[t+1])
        loss = loss / len(representations)

        self._is_fitted = True

        return loss

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.encoder(X)

