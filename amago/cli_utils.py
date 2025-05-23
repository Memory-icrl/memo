from argparse import ArgumentParser
from typing import Optional

import gin

import amago


def add_common_cli(parser: ArgumentParser) -> ArgumentParser:
    # extra gin configs
    parser.add_argument(
        "--configs",
        type=str,
        nargs="*",
        help="Extra `.gin` configuration files. These settings are usually added last in the examples and would overwrite the script's defaults.",
    )
    # basics
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--agent_type",
        type=str,
        default="agent",
        choices=["agent", "multitask"],
        help="Quick switch between default `agent.Agent` and `agent.MultiTaskAgent`. MultiTaskAgent is useful when training on mixed environments with multiple rewards functions.",
    )
    parser.add_argument(
        "--no_async",
        action="store_true",
        help="Run the 'parallel' actors in one thread. Saves resources when environments are already fast.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Turn off wandb logging (usually for debugging).",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Start training from an epoch checkpoint saved in a buffer with the same `--run_name`",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Give the run a name. Used for logging and the disk replay buffer. Experiments with the same run_name share the same replay buffer, but log separately.",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--buffer_dir",
        type=str,
        required=True,
        help="Path to disk location where replay buffer (and checkpoints) will be stored. Should probably be somewhere with lots of space...",
    )
    # trajectory encoder
    parser.add_argument(
        "--traj_encoder",
        choices=["ff", "transformer", "rnn", "mamba"],
        default="transformer",
        help="Quick switch between seq2seq models. (ff == feedforward (memory-free))",
    )
    parser.add_argument(
        "--memory_size",
        type=int,
        default=256,
        help="Model/token dimension for a Transformer; hidden state size for an RNN.",
    )
    parser.add_argument(
        "--memory_layers",
        type=int,
        default=3,
        help="Number of layers in the sequence model.",
    )
    # main learning schedule
    parser.add_argument(
        "--grads_per_epoch",
        type=int,
        default=1000,
        help="Gradient updates per training epoch.",
    )
    parser.add_argument(
        "--timesteps_per_epoch",
        type=int,
        default=1000,
        help="Timesteps of environment interaction per epoch *per actor*. The update:data ratio is defined by `grads_per_epoch / (timesteps_per_epoch * parallel_actors)`.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=20,
        help="How often (in epochs) to evaluate the agent on validation envs.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=50,
        help="How often (in epochs) to save an agent checkpoint.",
    )
    parser.add_argument(
        "--parallel_actors",
        type=int,
        default=12,
        help="Number of parallel environments (applies to training, validation, and testing).",
    )
    parser.add_argument(
        "--dset_max_size",
        type=int,
        default=20_000,
        help="Maximum size of the replay buffer (measured in trajectories, not timesteps).",
    )
    parser.add_argument(
        "--start_learning_at_epoch",
        type=int,
        default=0,
        help="Skip learning updates for this many epochs at the beginning of training (if worried about overfitting to a small dataset)",
    )
    parser.add_argument(
        "--slow_inference",
        action="store_true",
        help="Turn OFF fast-inference mode (key-value caching for Transformer, hidden state caching for RNN)",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "bf16"],
        default="no",
        help="Train in bf16 mixed precision (requires a compatible GPU). Make sure to select this option during `accelerate config`.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=12,
        help="Pytorch dataloader workers for loading trajectories from disk.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Training batch size (measured in trajectories, not timesteps).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["learn", "collect", "both"],
        help="Simple max-throughput async mode. Start the command with `--mode collect` 1+ times, and then start the same command with `--mode learner` in another terminal. Defaults to alternating collect/train steps.",
    )
    parser.add_argument(
        "--use_aux_future_pred",
        action="store_true",
        help="Use auxiliary future prediction loss.",
    )   
    parser.add_argument(
        "--aux_future_pred_loss_coeff",
        type=float,
        default=0.1,
        help="Coefficient for the auxiliary future prediction loss.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=8e-4,
        help="Learning rate for the optimizer.",
    )
    return parser


"""
Gin convenience functions.

Switch between the most common configurations without needing `.gin` config files.
"""


def switch_tstep_encoder(config: dict, arch: str, **kwargs):
    """
    Convenient way to switch between TstepEncoders without gin config files

    `kwargs` should be the names and new defaults of kwargs in the TstepEncoder we
    are using (either `FFTstepEncoder` if arch == "ff", or `CNNTstepEncoder` if arch == "cnn")
    """
    assert arch in ["ff", "cnn", "multimodalcnn", "multimodalposecnn"]
    if arch == "ff":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.FFTstepEncoder
        ff_config = "amago.nets.tstep_encoders.FFTstepEncoder"
        config.update({f"{ff_config}.{key}": val for key, val in kwargs.items()})
    elif arch == "cnn":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.CNNTstepEncoder
        cnn_config = "amago.nets.tstep_encoders.CNNTstepEncoder"
        config.update({f"{cnn_config}.{key}": val for key, val in kwargs.items()})
    elif arch == "multimodalcnn":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.MultimodalCNNTstepEncoder
        cnn_config = "amago.nets.tstep_encoders.MultimodalCNNTstepEncoder"
        config.update({f"{cnn_config}.{key}": val for key, val in kwargs.items()})
    elif arch == "multimodalposecnn":
        config[
            "amago.agent.Agent.tstep_encoder_Cls"
        ] = amago.nets.tstep_encoders.MultimodalPoseCNNTstepEncoder
        cnn_config = "amago.nets.tstep_encoders.MultimodalPoseCNNTstepEncoder"
        config.update({f"{cnn_config}.{key}": val for key, val in kwargs.items()})
    return config


def turn_off_goal_conditioning(config: dict):
    """
    Make the goal embedding network redundant when the environment is not goal-conditioned.
    This applies to most environments that do not use hindsight relabeling.
    """
    config.update(
        {
            "amago.nets.tstep_encoders.TstepEncoder.goal_emb_Cls": amago.nets.goal_embedders.FFGoalEmb,
            "amago.nets.goal_embedders.FFGoalEmb.zero_embedding": True,
            "amago.nets.goal_embedders.FFGoalEmb.goal_emb_dim": 1,
        }
    )
    return config


def switch_traj_encoder(config: dict, arch: str, memory_size: int, layers: int):
    """
    Convenient way to switch between TrajEncoders of different sizes without gin config files.
    """
    assert arch in ["ff", "rnn", "transformer", "mamba"]
    if arch == "transformer":
        tformer_config = "amago.nets.traj_encoders.TformerTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.TformerTrajEncoder,
                f"{tformer_config}.d_model": memory_size,
                f"{tformer_config}.d_ff": memory_size * 4,
                f"{tformer_config}.n_layers": layers,
            }
        )
    elif arch == "rnn":
        gru_config = "amago.nets.traj_encoders.GRUTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.GRUTrajEncoder,
                f"{gru_config}.n_layers": layers,
                f"{gru_config}.d_output": memory_size,
                f"{gru_config}.d_hidden": memory_size,
            }
        )

    elif arch == "ff":
        ff_config = "amago.nets.traj_encoders.FFTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.FFTrajEncoder,
                f"{ff_config}.d_model": memory_size,
                f"{ff_config}.n_layers": layers,
            }
        )
    elif arch == "mamba":
        mamba_config = "amago.nets.traj_encoders.MambaTrajEncoder"
        config.update(
            {
                "amago.agent.Agent.traj_encoder_Cls": amago.nets.traj_encoders.MambaTrajEncoder,
                f"{mamba_config}.d_model": memory_size,
                f"{mamba_config}.n_layers": layers,
            }
        )
    return config


def naive(config: dict, turn_off_fbc: bool = False):
    config.update(
        {
            "amago.nets.traj_encoders.TformerTrajEncoder.activation": "gelu",
            "amago.nets.actor_critic.NCritics.activation": "relu",
            "amago.nets.actor_critic.Actor.activation": "relu",
            "amago.nets.tstep_encoders.FFTstepEncoder.activation": "relu",
            "amago.nets.tstep_encoders.CNNTstepEncoder.activation": "relu",
            "amago.nets.tstep_encoders.MultimodalCNNTstepEncoder.activation": "relu",
            "amago.nets.tstep_encoders.MultimodalPoseCNNTstepEncoder.activation": "relu",
            "amago.nets.transformer.TransformerLayer.normformer_norms": False,
            "amago.nets.transformer.TransformerLayer.sigma_reparam": False,
            "amago.nets.transformer.AttentionLayer.sigma_reparam": False,
            "amago.nets.transformer.AttentionLayer.head_scaling": False,
            "amago.agent.Agent.num_critics": 2,
            "amago.agent.Agent.gamma": 0.99,
            "amago.agent.Agent.use_multigamma": False,
        }
    )

    if turn_off_fbc:
        config.update({"amago.agent.Agent.offline_coeff": 0.0})

    return config


def adaptive(config: dict):
    config.update(
        {
            "amago.nets.traj_encoders.TformerTrajEncoder.activation": "adaptive",
            "amago.nets.actor_critic.NCritics.activation": "adaptive",
            "amago.nets.actor_critic.Actor.activation": "adaptive",
            "amago.nets.tstep_encoders.FFTstepEncoder.activation": "adaptive",
            "amago.nets.tstep_encoders.CNNTstepEncoder.activation": "adaptive",
            "amago.nets.tstep_encoders.MultimodalCNNTstepEncoder.activation": "adaptive",
            "amago.nets.tstep_encoders.MultimodalPoseCNNTstepEncoder.activation": "adaptive",
        }
    )


def use_config(
    custom_params: dict, gin_configs: list[str] | None = None, finalize: bool = True
):
    """
    Bind all the gin parameters from real .gin configs (which the examples avoid using)
    and regular dictionaries.

    Use before training begins.
    """
    for param, val in custom_params.items():
        gin.bind_parameter(param, val)
    # override defaults with custom gin config files
    if gin_configs is not None:
        for config in gin_configs:
            gin.parse_config_file(config)
    if finalize:
        gin.finalize()


def create_experiment_from_cli(
    command_line_args,
    make_train_env: callable,
    make_val_env: callable,
    max_seq_len: int,
    traj_save_len: int,
    group_name: str,
    run_name: str,
    experiment_Cls=amago.Experiment,
    **extra_experiment_kwargs,
):
    cli = command_line_args

    experiment = experiment_Cls(
        agent_type=amago.agent.Agent
        if cli.agent_type == "agent"
        else amago.agent.MultiTaskAgent,
        make_train_env=make_train_env,
        make_val_env=make_val_env,
        max_seq_len=max_seq_len,
        traj_save_len=traj_save_len,
        dset_max_size=cli.dset_max_size,
        run_name=run_name,
        dset_name=run_name,
        dset_root=cli.buffer_dir,
        dloader_workers=cli.dloader_workers,
        log_to_wandb=not cli.no_log,
        wandb_group_name=group_name,
        batch_size=cli.batch_size,
        epochs=cli.epochs,
        parallel_actors=cli.parallel_actors,
        train_timesteps_per_epoch=cli.timesteps_per_epoch,
        train_grad_updates_per_epoch=cli.grads_per_epoch,
        start_learning_at_epoch=cli.start_learning_at_epoch,
        val_interval=cli.val_interval,
        ckpt_interval=cli.ckpt_interval,
        mixed_precision=cli.mixed_precision,
        fast_inference=not cli.slow_inference,
        async_envs=not cli.no_async,
        tformer_type=cli.tformer_type,
        summary_length=cli.summary_length,
        segment_length=cli.segment_length,
        detach_prob=cli.detach_prob,
        use_aux_future_pred=cli.use_aux_future_pred,
        aux_future_pred_loss_coeff=cli.aux_future_pred_loss_coeff,
        learning_rate=cli.learning_rate,
        **extra_experiment_kwargs,
    )

    return experiment


def make_experiment_learn_only(experiment: amago.Experiment) -> amago.Experiment:
    experiment.start_collecting_at_epoch = float("inf")
    experiment.train_timesteps_per_epoch = 0
    experiment.val_interval = 10
    experiment.val_timesteps_per_epoch = 0
    experiment.val_checks_per_epoch = 0
    experiment.parallel_actors = 2
    experiment.async_envs = False
    experiment.always_save_latest = True
    return experiment


def make_experiment_collect_only(experiment: amago.Experiment) -> amago.Experiment:
    experiment.start_collecting_at_epoch = 0
    experiment.start_learning_at_epoch = float("inf")
    experiment.train_grad_updates_per_epoch = 0
    experiment.val_checks_per_epoch = 0
    experiment.ckpt_interval = None
    experiment.always_save_latest = False
    experiment.always_load_latest = True
    # run "forever"; terminate manually (when learning process is done)
    experiment.epochs = 10_000_000
    experiment.dset_filter_pct = None
    return experiment


def switch_mode_load_ckpt(
    experiment: amago.Experiment, command_line_args
) -> amago.Experiment:
    cli = command_line_args
    if cli.mode == "collect":
        assert cli.trials == 1, "Async Mode breaks `trials` loop. Set `--trials = 1`"
        experiment = make_experiment_collect_only(experiment)
    elif cli.mode == "learn":
        assert cli.trials == 1, "Async Mode breaks `trials` loop. Set `--trials = 1`"
        experiment = make_experiment_learn_only(experiment)
        if cli.ckpt is not None:
            experiment.load_checkpoint(cli.ckpt)
    return experiment
