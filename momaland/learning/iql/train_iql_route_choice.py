"""Train IQL on the route choice problem domain, under linear scalarization, and save the results to a csv file."""

import argparse
import os
import random
from distutils.util import strtobool

import numpy as np
import pandas as pd

from momaland.learning.iql.iql import train
from momaland.learning.utils import mkdir_p


def compute_utlity(weights, rewards):
    """Compute the utility of a given action based on the weights and the rewards of the objectives."""
    return np.dot(weights, rewards)


def parse_args():
    """Argument parsing for hyperparameter optimization."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="moroute_choice_v0",
                        help="Environment ID, e.g. momaland/moroute_choice-v0 (legacy IDs also accepted)")
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument('--seed', type=int, default=1,
                        help="the seed of the experiment")
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--debug', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run in debug mode")
    parser.add_argument('--random', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="take random actions")
    parser.add_argument('--weights', type=float, nargs=2, default=[0.5, 0.5], )

    # Algorithm specific parameters
    parser.add_argument('--num-iterations', type=int, default=1000,
                        help="the number of training iterations")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="the learning rate")
    parser.add_argument('--alpha-decay', type=float, default=0.99,
                        help="the learning rate decay")
    parser.add_argument('--alpha-min', type=float, default=0.0,
                        help="the minimum learning rate")
    parser.add_argument('--epsilon', type=float, default=0.5,
                        help="the exploration rate")
    parser.add_argument('--epsilon-decay', type=float, default=0.99,
                        help="the exploration rate decay")
    parser.add_argument('--epsilon-min', type=float, default=0.0,
                        help="the minimum exploration rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount rate")

    # Environment specific parameters
    parser.add_argument('--problem-name', type=str, default="Braess_1_4200_10_c1",
                        help="the name of the problem")
    parser.add_argument('--num-agents', type=int, default=4200,
                        help="the number of agents")
    parser.add_argument('--toll-mode', type=str, default="mct",
                        help="the toll mode")

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()

    env_args = {"problem_name": args.problem_name, "num_agents": args.num_agents, "toll_mode": "mct"}

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
        metrics = train(args, args.weights, env_args)
        # Save the best rewards encountered (pareto front plot)
        best_rewards.append(metrics["metrics"]["best_reward"])

        # Save the results of the run
        ep = metrics["metrics"]["returned_episode_returns"][:, 0]
        scal_rew = metrics["metrics"]["returned_episode_returns"][:, 3]
        avg_tt = metrics["metrics"]["returned_episode_returns"][:, 4]

        run_df = pd.DataFrame({"episode": ep, "scal_rew": scal_rew, "avg_tt": avg_tt})
        df_list.append(run_df)

    # Concatenate the results of all runs
    df_total = pd.concat(df_list)

    # ---------------------- #
    # Saving Learning Curves #
    # ---------------------- #
    mkdir_p("momaland/learning/iql/results/route_choice")
    if not args.random:
        df_total.to_csv(
            f"momaland/learning/iql/results/route_choice/iql_{args.problem_name}_{args.weights[0]}_{args.weights[1]}.csv",
            index=False,
        )
    else:
        df_total.to_csv(f"momaland/learning/iql/results/route_choice/iql_{args.problem_name}_random.csv", index=False)
