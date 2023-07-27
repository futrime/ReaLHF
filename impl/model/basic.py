from typing import List, Union

import torch
import transformers

import api.model
import api.utils


def hf_model_factory(model_cls):

    def create_huggingface_model(
        name: str,
        model_name_or_path: str,
        disable_dropout: bool,
        generation_kwargs: dict,
        device: Union[str, torch.device],
    ):
        module = api.utils.create_hf_nn(model_cls,
                                        model_name_or_path,
                                        disable_dropout=disable_dropout,
                                        init_from_scratch=False,
                                        generation_kwargs=generation_kwargs)
        tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
        return api.model.Model(name, module, tokenizer, device)

    return create_huggingface_model


api.model.register_model("causal_lm", hf_model_factory(transformers.AutoModelForCausalLM))