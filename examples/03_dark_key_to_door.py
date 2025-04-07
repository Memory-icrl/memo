from argparse import ArgumentParser

import wandb

import amago
from amago.envs.builtin.gym_envs import GymEnv
from amago.envs.builtin.room_key_door import RoomKeyDoor
from amago.envs.env_utils import BilevelEpsilonGreedy
from amago.cli_utils import *
import torch
import random
import numpy as np
import ipdb

def add_cli(parser):
    parser.add_argument("--meta_horizon", type=int, default=500)
    parser.add_argument("--room_size", type=int, default=9)
    parser.add_argument("--episode_length", type=int, default=50)
    parser.add_argument("--tformer_type", type=str, default="tformer")
    parser.add_argument("--summary_length", type=int, default=0)
    parser.add_argument("--segment_length", type=int, default=0)
    parser.add_argument("--detach_prob", type=float, default=0.0)
    parser.add_argument("--expt_seed", type=int, default=0)
    parser.add_argument("--naive", action="store_true")

    return parser


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_cli(parser)
    add_cli(parser)
    args = parser.parse_args()

    # set seed manually
    random.seed(args.expt_seed)
    torch.manual_seed(args.expt_seed)
    np.random.seed(args.expt_seed)

    config = {"amago.agent.Agent.reward_multiplier": 100.0}

    turn_off_goal_conditioning(config)
    switch_tstep_encoder(config, arch="ff", n_layers=2, d_hidden=128, d_output=64)
    switch_traj_encoder(
        config,
        arch=args.traj_encoder,
        memory_size=args.memory_size,
        layers=args.memory_layers,
    )
    if args.naive:
        naive(config)
    use_config(config, args.configs)

    group_name = f"{args.run_name}_dark_key_door"
    for trial in range(args.trials):
        run_name = group_name + f"_trial_{trial}"
        """
        Meta-RL is the same as any other environment except that we reset
        the task over and over again until a fixed time limit. The easiest
        way to do this in AMAGO is to design/wrap the meta-RL gym Env so that
        `env.reset()` *picks a new task* ("hard reset") but create kwargs to 
        reset to the *same* task ("soft reset"). In this example `RoomKeyDoor.reset()` 
        randomly generates a new environment while `RoomKeyDoor.reset(new_task=False) 
        resets the agent to the same environment.

        For meta-RL by *episodes* (k-shot) instead of
        timesteps, see the metaworld example.
        """
        make_train_env = lambda: GymEnv(
            gym_env=RoomKeyDoor(
                size=args.room_size, max_episode_steps=args.episode_length
            ),
            env_name="Dark-Key-To-Door",
            horizon=args.meta_horizon,
            zero_shot=False,
            # env.reset() is called between rollouts (new tasks), while
            # env.reset(**soft_reset_kwargs) is called within meta-RL rollouts
            # (same task).
            soft_reset_kwargs={"new_task": False},
        )
        experiment = create_experiment_from_cli(
            args,
            make_train_env=make_train_env,
            make_val_env=make_train_env,
            max_seq_len=args.meta_horizon,
            traj_save_len=args.meta_horizon,
            group_name=group_name,
            run_name=run_name,
            val_timesteps_per_epoch=args.meta_horizon * 4,
            exploration_wrapper_Cls=BilevelEpsilonGreedy,
        )
        # make sur eit loads
        # ipdb.set_trace()
        # switch_mode_load_ckpt(experiment, args)
        experiment.start()
        # experiment.load_checkpoint(args.ckpt)
        # logs_val, returns_val, successes_val  = experiment.evaluate_val(plot_traj_rew=True)
        # print(f"Validation returns: {returns_val}")

        experiment.learn()                                                            ## swap it out
        
        logs, returns, successes = experiment.evaluate_test(make_train_env, timesteps=20_000, render=False)
        print(f"Test returns: {returns}")
        wandb.finish()
