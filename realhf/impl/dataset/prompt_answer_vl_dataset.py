"""Definition of PromptAnswerVLDataset."""

from typing import Callable, List, Optional, TypedDict, cast

import torch
import transformers

from realhf.api.core.base_processor import BaseProcessor
from realhf.api.core.data_api import (
    DatasetUtility,
    SequenceSample,
    load_shuffle_split_dataset,
    register_dataset,
)
from realhf.impl.dataset.base_dataset import BaseDataset


class PromptAnswerVLDatasetEntry(TypedDict):
    """A single entry in the JSON representation of PromptAnswerVLDataset."""

    id: int
    prompt: str
    answer: str
    images: List[str]


class PromptAnswerVLDataset(BaseDataset):
    """A dataset with vision-language prompts and corresponding answers. Usually used for SFT."""

    _dataset_utility: DatasetUtility
    _entries: List[PromptAnswerVLDatasetEntry]
    _max_length: int
    _pad_to_max_length: bool

    def __init__(  # pylint: disable=too-many-arguments
        self,
        util: DatasetUtility,
        max_length: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[
            Callable[[], List[PromptAnswerVLDatasetEntry]]
        ] = None,
        pad_to_max_length: bool = False,
    ):
        """Initializes the dataset.

        Args:
            util: The dataset utility.
            max_length: The maximum length of each sequence in the batch.
            dataset_path: Path to the dataset json/jsonl file.
            dataset_builder: A callable that returns the raw dataset. Alternative to dataset_path.
            pad_to_max_length: Whether to pad sequences to the maximum length.
        """

        self._dataset_utility = util
        self._max_length = max_length
        self._pad_to_max_length = pad_to_max_length

        self._entries = cast(
            List[PromptAnswerVLDatasetEntry],
            load_shuffle_split_dataset(util, dataset_path, dataset_builder),
        )

    @property
    def util(self) -> DatasetUtility:
        return self._dataset_utility

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> SequenceSample:
        entry = self._entries[idx]

        processor = cast(BaseProcessor, self._dataset_utility.tokenizer)

        eos_token = cast(str, processor.tokenizer.eos_token)

        sequence = entry["prompt"] + entry["answer"] + eos_token

        images = [
            transformers.image_utils.load_image(image_path)
            for image_path in entry["images"]
        ]

        sequence_feature = processor(text=sequence, images=images, return_tensors="pt")

        prompt_feature = processor(
            text=entry["prompt"], images=images, return_tensors="pt"
        )

        return SequenceSample.from_default(
            seqlens=[sequence_feature["input_ids"].shape[1]],
            ids=[entry["id"]],
            data={
                "packed_input_ids": cast(torch.Tensor, sequence_feature["input_ids"])
                .squeeze()
                .to(dtype=torch.int64),
                "prompt_mask": torch.cat(
                    [
                        torch.ones(
                            prompt_feature["input_ids"].shape[1],
                            dtype=torch.bool,
                        ),
                        torch.zeros(
                            sequence_feature["input_ids"].shape[1]
                            - prompt_feature["input_ids"].shape[1],
                            dtype=torch.bool,
                        ),
                    ]
                ),
            },
            metadata={
                "entry": [entry],
            },
        )


register_dataset("prompt_answer_vl", PromptAnswerVLDataset)
