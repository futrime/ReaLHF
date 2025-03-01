from io import BytesIO
from typing import TypedDict, cast

import faker
import pytest
import torch
import transformers
from PIL import Image

import realhf.base.constants as real_constants
import realhf.base.testing as real_testing
from realhf.api.core.base_processor import BaseProcessor
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.nn import real_llm_api
from realhf.impl.model.nn.real_llm_api import DuckModelOutput, ReaLModel
from tests.model.test_cpu_inference import maybe_prepare_cpu_env


@pytest.fixture(params=["qwen2_5_vl"], scope="session")
def model_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def model_config(model_name: str) -> ReaLModelConfig:
    return getattr(ReaLModel, f"make_{model_name}_config")()


@pytest.fixture
def model(model_config: ReaLModelConfig) -> ReaLModel:
    maybe_prepare_cpu_env(cast(int, model_config.n_positions))

    with real_constants.model_scope(cast(ModelName, real_testing.MODEL_NAME)):
        model = ReaLModel(model_config, dtype=torch.float32, device="cpu")
        real_llm_api.add_helper_functions(model)
        model.instantiate()
        model.eval()

    return model


@pytest.fixture(scope="session")
def processor() -> transformers.Qwen2_5_VLProcessor:
    processor = transformers.Qwen2_5_VLProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct"
    )

    return cast(transformers.Qwen2_5_VLProcessor, processor)


class TestData(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor


@pytest.fixture(scope="session")
def test_data(
    model_config: ReaLModelConfig,
    processor: transformers.Qwen2_5_VLProcessor,
) -> TestData:
    BATCH_SIZE = 10

    fake = faker.Faker()

    batch_feature = processor.__call__(
        text=[processor.image_token + fake.paragraph() for _ in range(BATCH_SIZE)],
        images=[Image.open(BytesIO(fake.image())) for _ in range(BATCH_SIZE)],
        max_length=model_config.n_positions,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": batch_feature["input_ids"],
        "attention_mask": batch_feature["attention_mask"],
        "pixel_values": batch_feature["pixel_values"],
        "image_grid_thw": batch_feature["image_grid_thw"],
    }


def test_cpu_inference(
    model: ReaLModel, model_config: ReaLModelConfig, test_data: TestData
):
    with (
        torch.no_grad(),
        real_constants.model_scope(cast(ModelName, real_testing.MODEL_NAME)),
    ):
        output: DuckModelOutput = model(
            input_ids=test_data["input_ids"],
            attention_mask=test_data["attention_mask"],
            pixel_values=test_data["pixel_values"],
            image_grid_thw=test_data["image_grid_thw"],
        )

        logits = output.logits

        pass
