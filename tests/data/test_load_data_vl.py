import uuid
from typing import List, Optional, Union, cast

import faker
import pytest
import torch
import transformers

from realhf.api.core import data_api
from realhf.api.core.base_processor import BaseProcessor
from realhf.api.core.config import DatasetAbstraction
from realhf.api.core.data_api import SequenceSample
from realhf.impl.dataset.prompt_answer_vl_dataset import PromptAnswerVLDatasetEntry


@pytest.fixture(scope="session")
def processor() -> BaseProcessor:
    processor = transformers.Qwen2_5_VLProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    return cast(BaseProcessor, processor)


@pytest.fixture(scope="session")
def raw_prompt_answer_vl_dataset(
    request: pytest.FixtureRequest,
    processor: BaseProcessor,
    tmp_path_factory: pytest.TempPathFactory,
) -> List[PromptAnswerVLDatasetEntry]:
    N_ENTRY = 10  # Number of entries in the dataset.

    fake = faker.Faker()
    temp_dir = tmp_path_factory.mktemp(raw_prompt_answer_vl_dataset.__name__)

    def create_entry(i: int) -> PromptAnswerVLDatasetEntry:
        image_bytes = fake.image()
        image_path = temp_dir / f"{i}.png"
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return {
            "id": i,
            "prompt": processor.image_token + fake.paragraph(),
            "answer": fake.paragraph(),
            "images": [str(image_path)],
        }

    dataset: List[PromptAnswerVLDatasetEntry] = [
        create_entry(i) for i in range(N_ENTRY)
    ]

    return dataset


@pytest.mark.parametrize("max_length", [128, 256, 1024])
def test_prompt_answer_vl_dataset(
    max_length: int,
    processor: BaseProcessor,
    raw_prompt_answer_vl_dataset: List[PromptAnswerVLDatasetEntry],
):
    import realhf.impl.dataset

    config = DatasetAbstraction(
        type_="prompt_answer_vl",
        args={
            "max_length": max_length,
            "dataset_builder": lambda: raw_prompt_answer_vl_dataset,
        },
    )

    _validate_dataset(config, processor)


def _validate_dataset(config: DatasetAbstraction, processor: BaseProcessor):
    dataset = data_api.make_dataset(
        config,
        seed=1,
        dp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=processor,
        experiment_name=uuid.uuid4().hex,
        trial_name=uuid.uuid4().hex,
    )

    dataloader = data_api.PackedDataLoader(dataset)

    for x in dataloader:
        assert isinstance(x, SequenceSample)
        assert x.data is not None

        for k, v in x.data.items():
            assert v is not None
            assert v.device == torch.device("cpu")

        bs = len(x.ids)

        for k, vs in x.seqlens.items():
            assert all(isinstance(v, list) for v in vs)
            assert all(all(isinstance(vv, int) for vv in v) for v in vs)

        assert len(x.ids) == len(set(x.ids))
        if x.metadata:
            for k, v in x.metadata.items():
                assert isinstance(v, list), k
        xs = x.split(bs)
        for xx in xs:
            if xx.metadata:
                for k, v in xx.metadata.items():
                    assert isinstance(v, list), k
                    assert len(v) == 1
