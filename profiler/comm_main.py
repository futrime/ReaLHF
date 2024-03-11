from typing import Dict, List, Optional
import argparse
import getpass
import os
import re

import profiler.experiments

from apps.main import _submit_workers
import api.config as config_package
import base.logging as logging
import base.name_resolve
import base.names
import scheduler.client
import system

logger = logging.getLogger("main", "system")

CONTROLLER_TIME_LIMIT = None
TRACE_TIMEOUT = (
    300  # Should be larger than TRACER_SAVE_INTERVAL_SECONDS defined in system/worker_base.py
)
EXPERIMENT_NAME = "profile_comm"
TRIAL_NAME = "profile_comm"


def main():
    trial_name = "profile_comm"
    expr_name = "profile_comm"
    experiment = config_package.make_experiment(expr_name)
    sched = scheduler.client.make(mode="slurm", expr_name=expr_name, trial_name=trial_name)

    setup = experiment.scheduling_setup()

    base_environs = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
        "WANDB_MODE": "disabled",
        "DLLM_MODE": "SLURM",
        "DLLM_TRACE": "0",
    }

    logger.info(f"Resetting name resolving repo...")
    try:
        base.name_resolve.clear_subtree(
            base.names.trial_root(experiment_name=expr_name, trial_name=trial_name))
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        raise e
    logger.info(f"Resetting name resolving repo... Done.")

    logger.info(f"Running configuration: {experiment.__class__.__name__}")

    # Schedule controller
    controller_type = "zmq"
    # For local_ray mode, the controller will start all remote workers.
    sched.submit_array(
        worker_type="ctl",
        cmd=scheduler.client.control_cmd(
            expr_name,
            trial_name,
            False,
            False,
            controller_type,
        ),
        count=1,
        cpu=1,
        gpu=0,
        mem=1024,
        env_vars=base_environs,
        container_image=setup.controller_image,
        time_limit=CONTROLLER_TIME_LIMIT,
    )

    workers_configs = ((k, getattr(setup, k)) for k in system.WORKER_TYPES)

    for name, scheduling_setup in workers_configs:
        if not isinstance(scheduling_setup, list):
            scheduling_setup = [scheduling_setup]
        # For local or slurm mode, launch all workers.
        # For ray mode, launch the ray cluster for all workers via slurm.
        _submit_workers(
            sched,
            expr_name,
            trial_name,
            False,
            name,
            scheduling_setup,
            base_environs,
            None,
            use_ray_cluster=False,
        )

    # timeout = None if not args.trace else TRACE_TIMEOUT  # run 5 mins to collect trace
    try:
        sched.wait(timeout=None)
    except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
        sched.stop_all()
        raise e


if __name__ == "__main__":
    main()
