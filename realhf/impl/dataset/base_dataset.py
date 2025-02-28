"""Definition of BaseDataset class."""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from realhf.api.core.data_api import DatasetUtility, SequenceSample


class BaseDataset(ABC, Dataset):
    """Base dataset class."""

    @property
    @abstractmethod
    def util(self) -> DatasetUtility:
        """Gets the dataset utility."""

    @abstractmethod
    def __len__(self) -> int:
        """Gets the size of the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> SequenceSample:
        """Gets a sample from the dataset.

        Args:
            idx: The index of the sample.

        Returns:
            The sample.
        """
