"""Train IQL on the Beach Problem Domain, under linear scalarization, and save the results to a csv file."""

import argparse
import random
import time
from distutils.util import strtobool

import numpy as np
import pandas as pd
from pf_bpd import fast_p_prune

from momaland.envs.beach.beach import (
    _global_capacity_reward,
    _global_mixture_reward,
    _local_capacity_reward,
    _local_mixture_reward,
)
from momaland.learning.iql.iql import train
from momaland.learning.utils import mkdir_p


def compute_normalization_constants(num_agents, sections, capacity, type_distribution, reward_scheme):
    """Compute the normalization constants for the given environment parameters.

    Args:
        num_agents (int): the number of agents
        sections (int): the number of sections
        capacity (int): the capacity of the sections
        type_distribution (list): the distribution of types
        reward_scheme (str): the reward scheme to use

    Returns:
        tuple: the normalization constants for capacity and mixture (min and max for each objective)
    """
    # Maximum global capacity: there are 'capacity' agents in each section, except one where all other agents are
    optimal_consumption = [capacity for _ in range(sections)]
    optimal_consumption[-1] = max(num_agents - ((sections - 1) * capacity), 0)
    max_cap_global = _global_capacity_reward([capacity] * sections, optimal_consumption)
    # Maximum local capacity is achieved when there are 'capacity' agents in the section
    max_cap_local = _local_capacity_reward(capacity, capacity)
    cap_min = 0.0
    cap_max = max_cap_local if reward_scheme == "individual" else max_cap_global

    #   Mixture
    # Maximum global mixture: one agent of each type in each section, except one where all other agents are
    num_F_agents = int(num_agents * type_distribution[0])
    num_M_agents = num_agents - num_F_agents
    remaining_F_agents = num_F_agents - (sections - 1)
    remaining_M_agents = num_M_agents - (sections - 1)
    types_per_section = [(1, 1) for _ in range(sections)]
    types_per_section[-1] = (remaining_F_agents, remaining_M_agents)
    max_mix_global = _global_mixture_reward(types_per_section)
    # Maximum local mixture is achieved when there is one agent of each type in the section
    max_mix_local = _local_mixture_reward([1, 1])
    mix_min = 0.0
    mix_max = max_mix_local if reward_scheme == "individual" else max_mix_global

    return cap_min, cap_max, mix_min, mix_max


def parse_args():
    """Argument parsing for hyperparameter optimization."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="mobeach_v0",
                        help="Environment ID, e.g. momaland/mobeach-v0 (legacy IDs also accepted)")
    parser.add_argument('--exp-name', type=str, default="exp1",
                        help="the name of this experiment")
    parser.add_argument('--seed', type=int, default=1,
                        help="the seed of the experiment")
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--random', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run with random actions")

    # Algorithm specific parameters
    parser.add_argument('--num-iterations', type=int, default=10000,
                        help="the number of training iterations")
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="the learning rate")
    parser.add_argument('--alpha-decay', type=float, default=1,
                        help="the learning rate decay")
    parser.add_argument('--alpha-min', type=float, default=0.,
                        help="the minimum learning rate")
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help="the exploration rate")
    parser.add_argument('--epsilon-decay', type=float, default=0.9999,
                        help="the exploration rate decay")
    parser.add_argument('--epsilon-min', type=float, default=0.0,
                        help="the minimum exploration rate")
    parser.add_argument('--gamma', type=float, default=0.9,
                        help="the discount rate")

    # Environment specific parameters
    parser.add_argument('--num-timesteps', type=int, default=1, help="the number of timesteps")
    parser.add_argument('--num-agents', type=int, default=50, )
    parser.add_argument('--type-distribution', type=float, nargs=2, default=[0.7, 0.3], )
    parser.add_argument('--position-distribution', type=float, nargs=5, default=[0., 0.5, 0., 0.5, 0.], )
    parser.add_argument('--sections', type=int, default=5, )
    parser.add_argument('--capacity', type=int, default=3, )
    parser.add_argument('--reward-scheme', type=str, default="individual", help="the reward scheme to use")

    args = parser.parse_args()
    args.time = time.time()

    return args


if __name__ == "__main__":
    args = parse_args()

    env_args = {
        "num_timesteps": args.num_timesteps,
        "num_agents": args.num_agents,
        "type_distribution": args.type_distribution,
        "position_distribution": args.position_distribution,
        "sections": args.sections,
        "capacity": args.capacity,
        "reward_mode": args.reward_scheme,
        # Normalization constants
        "local_constants": compute_normalization_constants(
            args.num_agents, args.sections, args.capacity, args.type_distribution, "individual"
        ),
        "global_constants": compute_normalization_constants(
            args.num_agents, args.sections, args.capacity, args.type_distribution, "team"
        ),
    }

    weights = [0.5, 0.5]

    # Keep track of the results of each run
    df_list = []
    # Keep track of the best rewards encountered
    best_rewards = []
    # Run the experiment args.runs times
    for i in range(args.runs):
        # Set the seed for reproducibility
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        # Training
        print(f"Running experiment {i + 1}/{args.runs}")
        metrics = train(args, weights, env_args)
        # Save the best rewards encountered (pareto front plot)
        best_rewards.append(metrics["metrics"]["best_reward"])
        # Save the results of the run
        ep = metrics["metrics"]["returned_episode_returns"][:, 0]
        scal_rew = metrics["metrics"]["returned_episode_returns"][:, 3]

        # Reshape the arrays and compute the mean of every 10 episodes
        ep_avg = np.arange(10, len(ep) + 1, 10)
        scal_rew_avg = scal_rew.reshape(-1, 10).mean(axis=1)

        # Save the results to a dataframe
        run_df = pd.DataFrame({"episode": ep_avg, "scal_rew": scal_rew_avg})
        df_list.append(run_df)

    # Concatenate the results of all runs
    df_total = pd.concat(df_list)

    # ---------------------- #
    # Saving Learning Curves #
    # ---------------------- #
    mkdir_p("momaland/learning/iql/results/beach/runs")
    if args.random:
        df_total.to_csv(f"momaland/learning/iql/results/beach/runs/BPD_{args.num_agents}_random.csv", index=False)
    else:
        df_total.to_csv(
            f"momaland/learning/iql/results/beach/runs/BPD_{args.num_agents}_{args.reward_scheme}.csv", index=False
        )

    # ------------------------------ #
    # Saving Non-Dominated Solutions #
    # ------------------------------ #
    mkdir_p("momaland/learning/iql/results/beach/nds")
    # Remove dominated solutions
    best_rewards = fast_p_prune(best_rewards)
    # Save remaining solutions to csv
    df = pd.DataFrame(best_rewards, columns=["Capacity", "Mixture"])
    # Save to csv
    if args.random:
        df.to_csv(f"momaland/learning/iql/results/beach/nds/BPD_{args.num_agents}_random.csv", index=False)
    else:
        df.to_csv(f"momaland/learning/iql/results/beach/nds/BPD_{args.num_agents}_{args.reward_scheme}.csv", index=False)
