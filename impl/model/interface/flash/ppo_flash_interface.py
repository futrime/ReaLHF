from typing import Dict, Optional, Tuple
import collections
import dataclasses
import itertools

import torch

from base.constants import data_parallel_group
from base.dataparallel import PackedParallelDataBroker
from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_engine.stream_pipe_engine import StreamPipeEngine
from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from impl.model.utils.functional import gather_packed_shifted_log_probs, masked_normalization
from impl.model.utils.save_load import save_hf_or_lora_model, save_pipeline_model
import api.huggingface
import api.model
import base.logging as logging
import impl.model.utils.ppo_functional as ppo_functional

try:
    from flash_attn.bert_padding import unpad_input
except ModuleNotFoundError:
    pass

logger = logging.getLogger("PackedPPOInterface")


def _ppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    packed_input_ids: torch.LongTensor,  # [tot_seqlen]
    cu_seqlens: torch.LongTensor,  # [bs+1]
    old_logp: torch.FloatTensor,  # [tot_seqlen-bs]
    ppo_loss_mask: torch.FloatTensor,  # [tot_seqlen-bs]
    advantages: torch.FloatTensor,  # [tot_seqlen-bs]
    kl_rewards: torch.FloatTensor,  # [tot_seqlen-bs]
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: int,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
    logits_mask: Optional[torch.BoolTensor] = None,  # [tot_seqlen, vocab_size]
    **kwargs,
) -> Tuple[torch.FloatTensor, Dict]:
    """Loss function for ppo actor step, all inputs should be splitted into pipeline micro batches,
    returns loss and logging stats.
    """
    if logits_mask is not None:
        logits.masked_fill_(logits_mask.logical_not_(),
                            torch.finfo(logits.dtype).min)  # inplace operation for logits mask
    new_logp = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids).float()

    new_logp = new_logp * ppo_loss_mask

    loss, loss_stat = ppo_functional.actor_loss_fn(
        logprobs=new_logp,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=ppo_loss_mask,
    )

    mean_ref_kl = (kl_rewards.detach() * ppo_loss_mask).sum() / ppo_loss_mask.sum()
    mean_ref_kl = api.huggingface.get_all_reduce_mean(mean_ref_kl, group=data_parallel_group())
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    importance_weight = loss_stat["importance_weight"]
    clip_ratio = loss_stat["clip_ratio"]
    if early_stop_imp_ratio is not None and importance_weight > early_stop_imp_ratio:
        logger.warning(f"Current importance ratio {importance_weight.item():.4f} is larger "
                       f"than early stop threshold {early_stop_imp_ratio}. Abandon this minibatch.")
        loss = loss * 0.0

    approx_kl = ((old_logp - new_logp).detach() * ppo_loss_mask).sum() / ppo_loss_mask.sum()

    stats = dict(
        ppo_approx_kl=approx_kl,
        cur_kl_ctl=torch.tensor(kl_adapter.value).to(approx_kl),
        actor_loss=loss.detach(),
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )

    if logits_mask is not None:
        stats["ignoring_logits_ratio"] = logits_mask.half().mean()  # inversed logits mask

    if (early_stop_kl is not None and approx_kl > early_stop_kl):
        logger.warning(f"Current approximate KL divergence {approx_kl.item():.4f} is larger "
                       f"than early stop threshold {early_stop_kl}. Abort actor update.")
        loss = loss * 0.0

    return loss, stats


@dataclasses.dataclass
class PackedActorInterface(api.model.ModelInterface):
    n_minibatches: int = 4

    generation_config: Optional[Dict] = None

    kl_ctl: float = 0.1

    adv_norm: bool = True
    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    enable_save: bool = True
    force_no_logits_mask: bool = False

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from impl.model.utils.modules import ExponentialRunningMeanStd, MovingAverageRunningMeanStd

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(beta=self.value_norm_beta, epsilon=self.value_norm_eps)
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    def save(self, model: api.model.Model, save_dir: str):
        if not self.enable_save:
            return
        if isinstance(model.module, DeepSpeedPipelineEngine):
            save_pipeline_model(model, save_dir)
        else:
            save_hf_or_lora_model(model, save_dir)

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))
        prompts: torch.LongTensor = data["prompts"]
        prompt_att_mask: torch.BoolTensor = data["prompt_att_mask"]
        bs, prompt_max_len = prompts.shape[:2]

        if isinstance(module, DeepSpeedPipelineEngine):
            packed_input_ids, _, cu_seqlens, _ = unpad_input(prompts, prompt_att_mask)

            res = module.generate(
                tokenizer=model.tokenizer,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                gconfig=GenerationConfig(**self.generation_config),
            )
            if res is None:
                return None

            gen_tokens, logprobs, logits_mask, *_ = res
        else:
            # unwrap deepspeed engine here
            module = module.module
            gen_res = module.generate(
                tokenizer=model.tokenizer,
                input_ids=prompts,
                attention_mask=prompt_att_mask,
                gconfig=GenerationConfig(**self.generation_config),
            )
            gen_tokens = gen_res.sequences
            logprobs = gen_res.scores
            logits_mask = gen_res.logits_mask

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(gen_tokens[:, -1] != pad_token_id)
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(gen_tokens != eos_token_id).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        prompt_lengths = prompt_att_mask.sum(1)

        # TODO: refactor the following whole bunch of sh*t.
        # Pack generated sequences and logprobs.
        prompts_list, prompt_log_probs_list, prompt_logits_mask_list = [], [], []
        gen_tokens_list, gen_log_probs_list, gen_logits_mask_list = [], [], []
        for i in range(bs):
            prompt_len, gen_len = prompt_lengths[i].item(), gen_lengths[i].item()

            # Prompts are left-padded. Besides, prompt_log_probs is one-step shorter than prompts.
            prompts_list.append(prompts[i, prompt_max_len - prompt_len:])
            prompt_log_probs_list.append(logprobs.new_zeros(prompt_len - 1))
            if logits_mask is not None:
                prompt_logits_mask_list.append(logits_mask.new_ones((prompt_len - 1, logits_mask.shape[-1])))

            # Generated tokens are right-padded.
            gen_tokens_list.append(gen_tokens[i, :gen_len])
            gen_log_probs_list.append(logprobs[i, :gen_len])
            if logits_mask is not None:
                gen_logits_mask_list.append(
                    torch.cat([logits_mask[i, :gen_len],
                               logits_mask.new_ones(1, logits_mask.shape[-1])]))

        # For complete sequences, EOS token is included. Otherwise the sequence may end with arbitrary token.
        # cu_seqlens marks the boundary of these sequences, no matter whether they are complete or not.
        packed_seq = torch.cat(list(itertools.chain.from_iterable(zip(prompts_list, gen_tokens_list))))
        seq_lengths = prompt_lengths + gen_lengths
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=seq_lengths.device),
             seq_lengths.cumsum(0)]).int()
        packed_logprobs = torch.cat(
            list(itertools.chain.from_iterable(zip(prompt_log_probs_list, gen_log_probs_list))))
        assert packed_seq.shape[0] == packed_logprobs.shape[0] + bs, (
            packed_seq.shape,
            packed_logprobs.shape,
            bs,
        )
        packed_logits_mask = None
        if gen_logits_mask_list and not self.force_no_logits_mask:
            packed_logits_mask = torch.cat(
                list(itertools.chain.from_iterable(zip(prompt_logits_mask_list, gen_logits_mask_list))))

        prompt_mask = zip(
            [torch.ones(plen, dtype=torch.bool, device=model.device) for plen in prompt_lengths],
            [torch.zeros(glen, dtype=torch.bool, device=model.device) for glen in gen_lengths],
        )
        prompt_mask = torch.cat(list(itertools.chain.from_iterable(prompt_mask)))

        res = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_seq=packed_seq,
            cu_seqlens=cu_seqlens,
            packed_logprobs=packed_logprobs,
            packed_logits_mask=packed_logits_mask.bool() if packed_logits_mask is not None else None,
            prompt_mask=prompt_mask,
        )
        return from_dict(res)

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        cu_seqlens = data["cu_seqlens"].int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        if isinstance(module, DeepSpeedPipelineEngine):
            res = module(packed_input_ids=data["packed_seq"], cu_seqlens=cu_seqlens)
            if res is None:
                return None
            logits = res
        else:
            res = module(packed_input_ids=data["packed_seq"], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            logits = res

        if "packed_logits_mask" in data and data["packed_logits_mask"] is not None:
            packed_logits_mask = data["packed_logits_mask"]
            logits.masked_fill_(packed_logits_mask.logical_not_(), torch.finfo(logits.dtype).min)
        logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, data["packed_seq"])
        return from_dict(dict(logprobs=logprobs))

    def train_step(self, model: api.model.Model, data_: NamedArray) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()
        data_ = recursive_apply(data_, lambda x: x.to(model.device))

        old_logp: torch.FloatTensor = data_["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = data_["packed_ref_logprobs"].float()
        prompt_mask = data_["prompt_mask"]
        cu_seqlens = data_["cu_seqlens"].int()
        reward_score = data_["rewards"].float()
        values = data_["values"].float()
        seq_no_eos_mask = data_["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
        shift_one_indices = torch.cat([
            torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
            for i in range(cu_seqlens.shape[0] - 1)
        ])
        loss_mask = loss_mask[shift_one_indices]

        ref_logp *= loss_mask
        old_logp *= loss_mask

        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)

        global_stats = dict(
            task_reward=reward_score.mean().detach(),
            kl_reward=(kl_rewards.detach() * loss_mask).sum().mean(),
            advantage=advantages.mean().detach(),
        )

        if self.adv_norm:
            advantages = masked_normalization(advantages, loss_mask)

        data_ = from_dict(
            dict(
                advantages=advantages,
                old_logp=old_logp,
                ppo_loss_mask=loss_mask,
                packed_seq=data_["packed_seq"],
                cu_seqlens=data_["cu_seqlens"].int(),
                kl_rewards=kl_rewards,
                logits_mask=data_["packed_logits_mask"] if "packed_logits_mask" in data_ else None,
            ))

        datas = PackedParallelDataBroker.scatter_to(data_, self.n_minibatches)
        # NOTE: We cannot randomly shuffle data here because data must the same shape across different pipeline stages.
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:
            cu_seqlens = data["cu_seqlens"]
            input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            logits_mask = data["packed_logits_mask"] if "packed_logits_mask" in data else None
            if isinstance(module, DeepSpeedPipelineEngine) or isinstance(module, StreamPipeEngine):
                module.set_version_steps(model.version.global_step)
                loss_fn_kwargs = dict(
                    input_lens=input_lens,  # used for partition
                    old_logp=data["old_logp"],
                    ppo_loss_mask=data["ppo_loss_mask"],
                    advantages=data["advantages"],
                    kl_rewards=data["kl_rewards"],
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                    logits_mask=logits_mask,
                )

                loss, stats = module.train_batch(
                    packed_input_ids=data["packed_seq"],
                    cu_seqlens=data["cu_seqlens"],
                    loss_fn=_ppo_actor_loss_from_model_outputs,
                    **loss_fn_kwargs,
                )
            else:
                max_seqlen = int(max(input_lens))
                output = module(
                    packed_input_ids=data["packed_seq"],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
                loss, stats = _ppo_actor_loss_from_model_outputs(
                    logits=output,
                    packed_input_ids=data["packed_seq"],
                    cu_seqlens=cu_seqlens,
                    old_logp=data["old_logp"],
                    ppo_loss_mask=data["ppo_loss_mask"],
                    advantages=data["advantages"],
                    kl_rewards=data["kl_rewards"],
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                    logits_mask=logits_mask,
                )

                module.backward(loss)
                module.step(lr_kwargs={"epoch": model.version.global_step})

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        if train_stats:
            train_stats: Dict[str, torch.Tensor] = dict(train_stats, **global_stats)
            for k, v in train_stats.items():
                v = v.detach() / self.n_minibatches
                train_stats[k] = v.item()

        return dict(train_stats)


def _ppo_critic_loss_from_model_outputs(
    new_values: torch.FloatTensor,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.LongTensor,
    values: torch.FloatTensor,
    ppo_loss_mask: torch.FloatTensor,
    returns: torch.FloatTensor,
    kl_rewards: torch.FloatTensor,
    value_eps_clip: float,
    kl_adapter: ppo_functional.KLController,
    rms=None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Dict]:
    leave_one_indices = torch.cat([
        torch.arange(cu_seqlens[i], cu_seqlens[i + 1] - 1, dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    new_values = new_values[leave_one_indices].squeeze(-1)
    values = values[leave_one_indices].squeeze(-1)

    loss, loss_stat = ppo_functional.critic_loss_fn(
        value=new_values,
        old_value=values,
        target_value=returns,
        value_eps_clip=value_eps_clip,
        loss_mask=ppo_loss_mask,
    )

    mean_ref_kl = (kl_rewards.detach() * ppo_loss_mask).sum() / ppo_loss_mask.sum()
    mean_ref_kl = api.huggingface.get_all_reduce_mean(mean_ref_kl, group=data_parallel_group())
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    clip_ratio = loss_stat["clip_ratio"]

    if rms is not None:
        denormalized_values = rms.denormalize(new_values)
    else:
        denormalized_values = new_values

    return loss, dict(
        value_loss=loss.detach(),
        value_clip_ratio=clip_ratio,
        normalized_values=new_values.mean().detach(),
        denormalized_values=denormalized_values.mean().detach(),
    )


@dataclasses.dataclass
class PackedCriticInterface(api.model.ModelInterface):
    n_minibatches: int = 4
    enable_save: bool = True
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from impl.model.utils.modules import ExponentialRunningMeanStd, MovingAverageRunningMeanStd

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(beta=self.value_norm_beta, epsilon=self.value_norm_eps)
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    def save(self, model: api.model.Model, save_dir: str):
        if not self.enable_save:
            return
        if isinstance(model.module, DeepSpeedPipelineEngine):
            save_pipeline_model(model, save_dir)
        else:
            save_hf_or_lora_model(model, save_dir)

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        cu_seqlens = data["cu_seqlens"].int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        if isinstance(module, DeepSpeedPipelineEngine):
            scores = module(packed_input_ids=data["packed_seq"], cu_seqlens=cu_seqlens)
            if scores is None:
                return None
        else:
            scores: torch.FloatTensor = module(packed_input_ids=data["packed_seq"],
                                               cu_seqlens=cu_seqlens,
                                               max_seqlen=max_seqlen)
        scores = scores.squeeze(-1)
        return from_dict(dict(scores=scores))

    def train_step(self, model: api.model.Model, data_: NamedArray) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()
        data_ = recursive_apply(data_, lambda x: x.to(model.device))

        old_logp = data_["packed_logprobs"].float()
        ref_logp = data_["packed_ref_logprobs"].float()
        prompt_mask = data_["prompt_mask"]
        cu_seqlens = data_["cu_seqlens"].int()
        reward_score = data_["rewards"].float()
        values = data_["values"].float()
        seq_no_eos_mask = data_["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)

        loss_mask = prompt_mask.logical_not()
        shift_one_indices = torch.cat([
            torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
            for i in range(cu_seqlens.shape[0] - 1)
        ])
        loss_mask = loss_mask[shift_one_indices]

        old_logp *= loss_mask
        ref_logp *= loss_mask

        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        _, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
            normalized_returns = self.rms.normalize(returns)
        else:
            normalized_returns = returns

        global_stats = dict(returns=returns.mean().detach())

        data_ = from_dict(
            dict(
                returns=normalized_returns,
                values=values,
                ppo_loss_mask=loss_mask,
                kl_rewards=kl_rewards,
                packed_seq=data_["packed_seq"],
                cu_seqlens=data_["cu_seqlens"],
            ))

        datas = PackedParallelDataBroker.scatter_to(data_, self.n_minibatches)
        # NOTE: We cannot randomly shuffle data here because data must the same shape across different pipeline stages.
        train_stats = collections.defaultdict(lambda: 0)
        for data in datas:
            input_lens = data["cu_seqlens"][1:] - data["cu_seqlens"][:-1]
            if isinstance(module, DeepSpeedPipelineEngine):
                module.set_version_steps(model.version.global_step)
                module.set_tokenizer(tokenizer)

                loss_kwargs = dict(
                    input_lens=input_lens,
                    values=data["values"],
                    ppo_loss_mask=data["ppo_loss_mask"],
                    returns=data["returns"],
                    kl_rewards=data["kl_rewards"],
                    value_eps_clip=self.value_eps_clip,
                    kl_adapter=self.kl_adapter,
                    rms=self.rms if self.value_norm else None,
                )

                loss, stats = module.train_batch(
                    packed_input_ids=data["packed_seq"],
                    cu_seqlens=data["cu_seqlens"],
                    loss_fn=_ppo_critic_loss_from_model_outputs,
                    **loss_kwargs,
                )
            else:
                max_seqlen = int(max(input_lens))
                new_values: torch.FloatTensor = module(packed_input_ids=data["packed_seq"],
                                                       cu_seqlens=data["cu_seqlens"],
                                                       max_seqlen=max_seqlen).float()

                loss, stats = _ppo_critic_loss_from_model_outputs(
                    new_values=new_values,
                    packed_input_ids=data["packed_seq"],
                    cu_seqlens=data["cu_seqlens"],
                    values=data["values"],
                    ppo_loss_mask=data["ppo_loss_mask"],
                    returns=data["returns"],
                    kl_rewards=data["kl_rewards"],
                    value_eps_clip=self.value_eps_clip,
                    kl_adapter=self.kl_adapter,
                    rms=self.rms if self.value_norm else None,
                )

                module.backward(loss)
                module.step(lr_kwargs={"epoch": model.version.global_step})

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        if train_stats:
            train_stats: Dict[str, torch.Tensor] = dict(train_stats, **global_stats)
            for k, v in train_stats.items():
                v = v.detach() / self.n_minibatches
                train_stats[k] = v.item()

        return dict(train_stats)


api.model.register_interface("flash_actor", PackedActorInterface)
api.model.register_interface("flash_critic", PackedCriticInterface)
