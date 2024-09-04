import os
import numpy as np
from datetime import datetime
import torch

import omni.isaac.core
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    # Initialize the Isaac Sim stage
    stage = omni.usd.get_context().get_stage()
    stage.LoadStage(os.path.join(LEGGED_GYM_ROOT_DIR, "path_to_your_stage.usd"))

    # Prepare the environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Prepare the algorithm runner
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Run the learning algorithm
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.headless = False  # Make sure to set this according to your needs
    train(args)
