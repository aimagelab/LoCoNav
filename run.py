#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random

import numpy as np
import torch

from habitat_baselines.common.baseline_registry import baseline_registry
from occant_baselines.config.default import get_config

from torch import distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def init_distributed(world_size, rank):
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '4242'

    distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)


def run_exp(exp_config: str, run_type: str, world_size, rank, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    config.defrost()
    config.WORLD_SIZE = world_size
    config.RANK = rank
    config.DISTRIBUTED = True
    config.freeze()

    init_distributed(world_size, rank)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


if __name__ == "__main__":
    main()
