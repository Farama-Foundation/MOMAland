"""Implementation of stateless independent Q-learners. Implemented for the multi-objective route choice game and beach domain."""

import random

import numpy as np
from pettingzoo import make

from momaland.learning.iql.tabular_bpd import TabularMOBeachDomainWrapper
from momaland.utils.registration import normalize_env_id


def compute_utility(weights, rewards):
    """Compute the utility of a given action based on the weights and the rewards of the objectives.

    Args:
        weights: the weights for the objectives
        rewards: the rewards for the objectives

    Returns:
        float: the utility of the action
    """
    return np.dot(weights, rewards)


class QAgent:
    """Q-learning agent."""

    def __init__(self, agent_id, n_states, n_actions):
        """Initialize the Q-agent."""
        self.agent_id = agent_id
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_values = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state, epsilon):
        """Epsilon-greedy action selection.

        Choose the action with the highest Q-value with probability 1-epsilon otherwise choose a random action.
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q_values[state])

    def update(self, state, new_state, action, alpha, gamma, reward, done):
        """Update the Q-values of the agent based on the chosen action and the new state."""
        # Retrieve the current Q-value
        q_value = self.q_values[state][action]
        # Compute the next max Q-value
        next_max_q_value = 0.0 if done else np.max(self.q_values[new_state])
        self.q_values[state][action] = (1 - alpha) * q_value + alpha * (reward + gamma * next_max_q_value)


def train(args, weights, env_args):
    """IQL scalarizing the vector reward using weights and weighted sum."""
    # Environment Initialization
    if normalize_env_id(args.env_id) == "momaland/mobeach-v0":
        # Use the tabular version of the MO-Beach domain with IQL
        env = TabularMOBeachDomainWrapper(**env_args)
    else:
        env = make("parallel", normalize_env_id(args.env_id), **env_args)
    obs, infos = env.reset()

    agents = {
        QAgent(agent_id, env.observation_space(env.agents[0]).n, env.action_spaces[agent_id].n)
        for agent_id in env.possible_agents
    }

    # Algorithm specific parameters
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min
    alpha = args.alpha
    alpha_decay = args.alpha_decay
    alpha_min = args.alpha_min
    gamma = args.gamma

    episode_returns = []
    # keep track of the best reward encountered
    best_reward = np.array([-np.inf, -np.inf])
    best_reward_scal = -np.inf

    for current_iter in range(args.num_iterations):
        # Get the actions from the agents
        if args.random:
            actions = {q_agent.agent_id: random.choice(range(env.action_spaces[q_agent.agent_id].n)) for q_agent in agents}
        else:
            actions = {q_agent.agent_id: q_agent.act(obs[q_agent.agent_id], epsilon) for q_agent in agents}

        # Update the exploration rate
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        else:
            epsilon = epsilon_min

        new_obs, rew, terminateds, truncations, infos = env.step(actions)

        # Check if all agents are terminated (in a stateless setting, this is always the case)
        # TODO: Handle truncation and termination difference
        terminated = np.logical_or(
            np.any(np.array(list(terminateds.values())), axis=-1), np.any(np.array(list(truncations.values())), axis=-1)
        )

        # Update the Q-values of the agents
        for agent in agents:
            agent.update(
                obs[agent.agent_id],
                new_obs[agent.agent_id],
                actions[agent.agent_id],
                alpha,
                gamma,
                compute_utility(weights, rew[agent.agent_id]),
                terminated,
            )

        if env.metadata["name"] == "mobeach_v0":
            # MO-Beach domain reports the global rewards in the info dict, regardless of the reward scheme
            avg_obj1 = infos[env.possible_agents[0]]["g_cap"]
            avg_obj2 = infos[env.possible_agents[0]]["g_mix"]

            avg_obj1_norm = infos[env.possible_agents[0]]["g_cap_norm"]
            avg_obj2_norm = infos[env.possible_agents[0]]["g_mix_norm"]

            # Keep track of best reward during training
            new_reward = np.array([avg_obj1, avg_obj2])
            scal_rew = compute_utility(weights, np.array([avg_obj1_norm, avg_obj2_norm]))
            episode_returns.append((current_iter, avg_obj1_norm, avg_obj2_norm, scal_rew))
        elif env.metadata["name"] == "moroute_choice_v0":
            avg_obj1 = np.mean(np.array(list(rew.values()))[:, 0])
            avg_obj2 = np.mean(np.array(list(rew.values()))[:, 1])
            scal_rew = compute_utility(weights, np.array([avg_obj1, avg_obj2]))
            new_reward = np.array([avg_obj1, avg_obj2])
            avg_tt = env.avg_tt
            episode_returns.append((current_iter, avg_obj1, avg_obj2, scal_rew, avg_tt))
        else:
            avg_obj1 = np.mean(np.array(list(rew.values()))[:, 0])
            avg_obj2 = np.mean(np.array(list(rew.values()))[:, 1])
            scal_rew = compute_utility(weights, np.array([avg_obj1, avg_obj2]))
            new_reward = np.array([avg_obj1, avg_obj2])
            episode_returns.append((current_iter, avg_obj1, avg_obj2, scal_rew))

        if scal_rew > best_reward_scal:
            best_reward = new_reward
            best_reward_scal = scal_rew

        # Update the learning rate
        if alpha > alpha_min:
            alpha *= alpha_decay
        else:
            alpha = alpha_min

        # In case of termination, reset the environment
        if terminated:
            env.reset()

    metric = {"returned_episode_returns": np.array(episode_returns), "best_reward": best_reward}
    return {"metrics": metric}
