"""Definition of BaseProcessor class."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from PIL.Image import Image
from transformers import BatchFeature, PreTrainedTokenizerBase, ProcessorMixin


class BaseProcessor(ABC, ProcessorMixin):
    """Base processor class."""

    tokenizer: PreTrainedTokenizerBase

    @abstractmethod
    def __call__(
        self,
        *,
        text: Union[str, List[str]],
        images: Optional[List[Image]] = None,
        videos: Optional[List[List[Image]]] = None,
    ) -> BatchFeature:
        """Processes the input data.

        Args:
            text: The text.
            images: The images.
            videos: The videos.

        Returns:
            The processed data with the following fields:
            - input_ids: The input IDs.
            - attention_mask: The attention mask.
            - pixel_values: The pixel values. Only returned when images is not None.
            - pixel_values_videos: The pixel values of videos. Only returned when videos is not
              None.
            - image_grid_thw: The image 3D grid in LLM. Only returned when images is not None.
            - video_grid_thw: The video 3D grid in LLM. Only returned when videos is not None.
        """
