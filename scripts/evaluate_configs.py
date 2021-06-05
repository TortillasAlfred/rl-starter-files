import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
import utils
import numpy as np
from collections import defaultdict
import pickle
import os


def plot_return_distributions(returns):
    # NEW: plot PDF and CDF of returns

    for leader in ["policy", "adversary"]:
        for config in returns:
            if config[0] != leader:
                continue

            returns_config = returns[config]

            # CDF
            plot = sns.ecdfplot(data=returns_config, label=f"alpha={config[1]}")
            plot.set(
                xlim=(0, 1), title="Returns CDF", xlabel="Return", ylabel="Percentile"
            )

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

        for config in returns:
            if config[0] != leader:
                continue

            returns_config = returns[config]

            # PDF
            plot = sns.kdeplot(
                returns_config, fill=True, alpha=0.5, label=f"alpha={config[1]}"
            )
            plot.set(title="Return Distribution", xlabel="Return")

        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


def get_returns_for_config(args):
    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        # Overrode the old flexible env loading to only load our stochastic env
        # envs.append(utils.make_env(args.env, args.seed + 10000 * i))
        envs.append(
            utils.get_stochastic_env(seed=args.seed + 10000 * i, delta=args.delta)
        )
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    policy_acmodel = utils.ACModel(envs[0].observation_space, envs[0].action_space)
    agent = utils.Agent(
        policy_acmodel,
        envs[0].observation_space,
        model_dir,
        device=device,
        argmax=True,
        num_envs=args.procs,
        pretrained=True,
    )
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    return logs["return_per_episode"]


def evaluate_configs(args):
    configs = [
        ("policy", 0.1, 4),
        ("policy", 0.25, 4),
        ("policy", 0.5, 4),
        ("policy", 0.9, 4),
        ("policy", 0.95, 4),
        ("policy", 1.0, 0),
        ("adversary", 0.1, 8),
        ("adversary", 0.25, 8),
        ("adversary", 0.5, 8),
        ("adversary", 0.9, 8),
        ("adversary", 0.95, 8),
    ]

    if os.path.isfile("config_returns.pck"):
        with open("config_returns.pck", "rb") as f:
            config_returns = pickle.load(f)
    else:
        config_returns = defaultdict(list)

        for seed in range(2):
            for leader, target_quantile, n_steps_follower in configs:
                args.model = f"leader={leader}_alpha={target_quantile}_K={n_steps_follower}_seed={seed}"

                returns = get_returns_for_config(args)
                config_returns[(leader, target_quantile)].extend(returns)

        with open("config_returns.pck", "wb") as f:
            pickle.dump(config_returns, f)

    plot_return_distributions(config_returns)


if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="number of episodes of evaluation (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--procs", type=int, default=16, help="number of processes (default: 16)"
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="action with highest probability is selected",
    )
    parser.add_argument(
        "--worst-episodes-to-show",
        type=int,
        default=10,
        help="how many worst episodes to show",
    )
    parser.add_argument(
        "--memory", action="store_true", default=False, help="add a LSTM to the model"
    )
    parser.add_argument(
        "--text", action="store_true", default=False, help="add a GRU to the model"
    )
    parser.add_argument(
        "--delta", type=float, default=0.05, help="env stochasticity (default: 0.05)"
    )

    args = parser.parse_args()

    np.seterr(all="ignore")

    evaluate_configs(args)
