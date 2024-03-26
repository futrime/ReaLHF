from collections import defaultdict
from typing import Dict
import gc
import itertools
import json
import multiprocessing as mp
import os
import queue
import socket
import time

from deepspeed.accelerator import get_accelerator
from flash_attn.bert_padding import unpad_input
import colorama
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from profiler.comm import ProfileCommunication
from profiler.engine import ProfileEngine
from profiler.utils import random_sample

from api.config.config_flash_model import FlashMQATConfig
from base.constants import LOG_ROOT
from base.monitor import gpu_utilization_monitor, time_mark
from base.topology import ParallelGrid
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
# from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
import api.config.config_system as config_package
import api.config.dfg
import api.data
import api.model
import base.constants
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.numpy_utils
import base.seeding as seeding
import base.timeutil
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

# Register all implemented datasets and models.
import impl.model  # isort:skip
import impl.dataset  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")


class ProfileCompelte(Exception):

    def __init__(self, message):
        disclaimer = (colorama.Fore.GREEN + "\033[1m" +
                      "<This is not an error. It is just a way to stop the experiment.> ")
        super().__init__(disclaimer + colorama.Style.RESET_ALL + colorama.Fore.YELLOW +
                         colorama.Style.BRIGHT + "\033[1m" + message + colorama.Style.RESET_ALL)


class ProfileWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)
        self.config = None
        self.model_name = None

        self.__ddp_env_resolved = False

    def _configure(self, cfg: config_package.ProfileWorker):
        self.config = cfg
        self.model_name = "profile"

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.dump_root = os.path.join(LOG_ROOT, self.__experiment_name, self.__trial_name)
        os.makedirs(self.dump_root, exist_ok=True)

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        # NOTE: We include master worker in the process group, so the global rank is model_worker_index + 1
        gpu_utils.reveal_ddp_identity_single_model(self.__experiment_name, self.__trial_name, self.model_name,
                                                   self.__worker_index)
        self.__ddp_env_resolved = False

        r = self.config.worker_info

        self.model_config = cfg.model
        self.interface_config = cfg.interface
        self.backend_config = cfg.backend
        self.rpcs = cfg.rpcs

        self.bs_list = cfg.bs_list
        self.seq_len_list = cfg.seq_len_list
        # self.gen_tokens_list = cfg.gen_tokens_list

        if self.bs_list is None:
            self.bs_list = [32, 64, 128, 256]
        if self.seq_len_list is None:
            self.seq_len_list = [128, 256, 512]
        # if self.gen_tokens_list is None:
        #     self.gen_tokens_list = [128]

        self.profile_rounds = cfg.profile_rounds
        self.warmup_rounds = cfg.warmup_rounds
        # self.current_profile_round = 0
        self.profile_start = None
        self.num_rpcs = len(self.rpcs)

        self.profile_communication = cfg.profile_communication
        self.profile_rpc = cfg.profile_rpc
        self.stats = defaultdict(list)
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        # base.constants.set_model_name(self.config.model_name)

        self.__pg_info = gpu_utils.setup_ddp_single_model(
            self.__experiment_name,
            self.__trial_name,
            self.model_name,
            self.__worker_index,
        )
        base.constants.set_parallelism_group(
            self.model_name,
            dist.group.WORLD,
        )
        base.constants.set_experiment_trial_names(self.__experiment_name, self.__trial_name)

        deepspeed.init_distributed()

        topo = self.config.topo
        grid = ParallelGrid(
            topology=topo,
            process_group=dist.group.WORLD,
        )
        base.constants.set_grid(self.model_name, grid)
        base.constants.set_rank_mapping(self.model_name, topo)

        logger.info(f"SetUp Information - Model worker index {self.__worker_index}"
                    f' type "{self.model_name}" located at '
                    f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        # if self.config.backend.type_ in ["ds_train", "ds_inference"]:
        self.logger.info("deepspeed init distributed on model worker")
        self.__device = torch.device("cuda:0")

        self.__gpu_util_mp = mp.Process(target=gpu_utilization_monitor,
                                        args=(self.__pg_info.local_gpu_id, 7200))
        self.__gpu_util_mp.start()

        if self.profile_communication:
            self.__profile_comm = ProfileCommunication("comm", self.__device, self.__pg_info.local_gpu_id,
                                                       self.__pg_info.global_rank, self.__pg_info.world_size)

        with base.constants.model_scope(self.model_name):
            base.constants.set_max_seqlen(max(self.seq_len_list))
            self.__interface = api.model.make_interface(self.interface_config)
            self.__backend = api.model.make_backend(self.backend_config)

            # assert isinstance(self.__model.module, FlashMQATModel)
            self.__engine = None

    def __reinit_backend(self, bs, seq_len):
        with base.constants.model_scope(self.model_name):
            self.__model = api.model.make_model(
                self.model_config,
                name=self.model_name,
                device=self.__device,
            )
            if isinstance(self.__model.module, FlashMQATModel):
                self.__model.module.instantiate()

            self.__nn_config = self.__model.module.config
            # print(self.__nn_config)
            self.__vocab_size = self.__nn_config.vocab_size

            bs_per_device = bs // base.constants.data_parallel_world_size()
            ft_spec = api.model.FinetuneSpec(10, 100, 10, bs_per_device, seq_len)
            self.__model = self.__backend.initialize(self.__model, ft_spec)
            self.__engine = self.__model.module

    def __run_model_function_call(self, rpc: api.config.dfg.ModelRPC, bs, seq_len) -> worker_base.PollResult:
        with base.constants.model_scope(self.model_name):
            # initialize
            func_name = rpc.interface_type.value
            data = random_sample(bs, seq_len, self.__vocab_size)
            stats_key = f"{rpc.name}|{bs}|{seq_len}"
            logger.info(f"Running model function call: {func_name} "
                        f"for rpc {stats_key}.")
            func = getattr(self.__interface, func_name)
            for i in range(self.warmup_rounds):
                func(self.__model, data)
                logger.info(f"{stats_key} warm up round {i} done")

            st = time.monotonic()
            for _ in range(self.profile_rounds):
                rt = time.monotonic()
                func(self.__model, data)
                self.stats[stats_key].append(time.monotonic() - rt)
            logger.info(f"Running {self.profile_rounds} model function call: {func_name} "
                        f"for rpc {stats_key} done, time cost = {time.monotonic() - st}, "
                        f"avg time cost per round = {np.mean(self.stats[stats_key])} s.")
            # instruction profiles
            # self.__interface.inference(self.__model, data)
            # self.__interface.train_step(self.__model, data)
            # self.__interface.generate(self.__model, data)
        return worker_base.PollResult(sample_count=1, batch_count=1)

    def _poll(self):
        try:
            if not self.__ddp_env_resolved:
                self.__lazy_setup()
                self.__ddp_env_resolved = True

            if self.profile_start is None:
                self.profile_start = time.monotonic()

            r = worker_base.PollResult(0, 0)
            if self.profile_rpc:
                bs_seq_len = itertools.product(self.bs_list, self.seq_len_list)
                bs_seq_len = sorted(bs_seq_len, key=lambda x: x[0] * x[1])
                for bs, seq_len in bs_seq_len:
                    self.__reinit_backend(bs, seq_len)
                    for rpc in self.rpcs:
                        # data = random_sample(bs, seq_len, self.__vocab_size)
                        r = self.__run_model_function_call(rpc, bs, seq_len)

                    # dump stats
                    with open(os.path.join(self.dump_root, "rpc_profile_stats.json"), "w") as f:
                        json.dump(self.stats, f)

                if r.batch_count > 0:
                    # following huggingface trl # ALWAYS COST 0.3+ SEC
                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()

            # if self.config.profile_model_function_call:
            #     self.__profile_comm.profile_comm()

            if self.profile_communication:
                for _ in range(self.warmup_rounds):
                    self.__profile_comm.profile_comm()
                self.__profile_comm.reset_stats()
                for _ in range(self.profile_rounds):
                    self.__profile_comm.profile_comm()
                self.__profile_comm.print_stats()
                self.__profile_comm.dump_stats()

            raise ProfileCompelte(f"Profile rounds {self.warmup_rounds + self.profile_rounds} complete !!! "
                                  f"total time cost {time.monotonic() - self.profile_start} s.")
        except Exception as e:
            self.__gpu_util_mp.terminate()
            self.__gpu_util_mp.join()
            raise e

        # blogger.debug("Current profile round done: {}".format(self.current_profile_round))
        # if self.current_profile_round <= self.warmup_rounds:
        #     # if self.config.profile_communication:
        #     #     self.__profile_comm.reset_stats()
        #     logger.info("Warmup round {} done.".format(self.current_profile_round))

        return r
