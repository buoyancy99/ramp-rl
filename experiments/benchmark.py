import os
import argparse
import wandb
import pathlib

from environments import env_episode_lens
from experiments.mpc_experiment import run_mpc_experiment
from experiments.sf_experiment import run_sf_experiment


os.environ["WANDB_START_METHOD"] = "thread"
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "dryrun")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--steps", type=int, default=32000)
    parser.add_argument("--load", action="store_true")

    # MPC (ours) specific arguments
    parser.add_argument("--eval_policy", type=str, choices=["online", "random", "expert"], default="online")
    parser.add_argument("--basis_type", type=str, choices=["poly", "rand", "none", "learned"], default="rand")
    parser.add_argument("--basis_dim", type=int, default=2048)
    parser.add_argument("--num_offline_epoch", type=int, default=4)
    parser.add_argument("--mpc_samples", type=int, default=1024)
    parser.add_argument("--mpc_horizon", type=int, default=10)
    parser.add_argument("--reweight", type=float, default=-1.0)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--disagreement_coef", type=float, default=1.0)
    parser.add_argument("--validation_ratio", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--vf_every", type=int, default=1600)
    parser.add_argument("--finetune_every", type=int, default=800)
    parser.add_argument("--finetune_epoches", type=int, default=1)

    # CQL specific
    parser.add_argument("--relabel_frac", type=float, default=0.5)

    # Debug arguments
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--log_reward_error", action="store_true")
    parser.add_argument("--log_q_error", action="store_true")
    parser.add_argument("--vis_offline_lsq", action="store_true")
    parser.add_argument("--planner", type=str, default="random_shooting")
    parser.add_argument("--cem_iters", type=int, default=10)
    parser.add_argument("--cem_k", type=int, default=5)
    parser.add_argument("--mppi_temp", type=float, default=10)

    return parser.parse_args()


experiment_registry = dict(
    mpc=run_mpc_experiment,
    sf=run_sf_experiment,
)


if __name__ == "__main__":
    args = get_args()
    if args.env_id not in env_episode_lens:
        print("Invalid environment! Quit")
    else:
        wandb.init(
            project="ramp",
            entity=os.environ.get("MY_WANDB_ID", None),
            group=args.env_id,
            job_type=args.algo,
            config=args,
        )

        wandb.config.update({"num_episodes": args.steps // env_episode_lens[args.env_id]})
        wandb.config.update({"debug_tag": 1})

        pathlib.Path(os.path.join("ckpts", args.env_id)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join("buffer", args.env_id)).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join("data", args.env_id)).mkdir(parents=True, exist_ok=True)
        print("\n\n")
        print("=" * 60)
        print(f"Benchmarking algo {wandb.config['algo']}, env {wandb.config['env_id']}, seed {wandb.config['seed']}")
        print("=" * 60)
        experiment_registry[args.algo](wandb.config)
