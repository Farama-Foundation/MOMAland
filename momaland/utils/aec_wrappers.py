"""Various wrappers for AEC MO environments."""

from typing import Optional

import numpy as np
from gymnasium.wrappers.utils import RunningMeanStd
from pettingzoo.utils.wrappers.base import BaseWrapper


class RecordEpisodeStatistics(BaseWrapper):
    """This wrapper will record episode statistics and print them at the end of each episode."""

    def __init__(self, env):
        """This wrapper will record episode statistics and print them at the end of each episode.

        Args:
            env (env): The environment to apply the wrapper
        """
        BaseWrapper.__init__(self, env)
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.possible_agents}

    def last(self, observe: bool = True):
        """Receives the latest observation from the environment, recording episode statistics."""
        obs, rews, terminated, truncated, infos = super().last(observe=observe)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] += rews
            self.episode_lengths[agent] += 1
        if terminated or truncated:
            infos["episode"] = {
                "r": self.episode_rewards,
                "l": self.episode_lengths,
            }
        return obs, rews, terminated, truncated, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resets the environment and the episode statistics."""
        super().reset(seed, options)
        for agent in self.env.possible_agents:
            self.episode_rewards[agent] = 0
            self.episode_lengths[agent] = 0


class LinearizeReward(BaseWrapper):
    """Convert MO reward vector into scalar SO reward value.

    `weights` represents the weights of each objective in the reward vector space for each agent.

    Example:
        >>> weights = {"agent_0": np.array([0.1, 0.9]), "agent_1": np.array([0.2, 0.8]}
        ... env = LinearizeReward(env, weights)
    """

    def __init__(self, env, weights: dict):
        """Reward linearization class initializer.

        Args:
            env: base env to add the wrapper on.
            weights: a dict where keys are agents and values are vectors representing the weights of their rewards.
        """
        self.weights = weights
        super().__init__(env)

    def last(self, observe: bool = True):
        """Returns a reward scalar from the reward vector."""
        observation, rewards, termination, truncation, info = self.env.last(observe=observe)
        if self.env.agent_selection in list(self.weights.keys()):
            rewards = np.dot(rewards, self.weights[self.env.agent_selection])
        return observation, rewards, termination, truncation, info


class NormalizeReward(BaseWrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.

    Example:
        >>> for agent in env.possible_agents:
        ...     for idx in range(env.reward_space(agent).shape[0]):
        ...         env = AECWrappers.NormalizeReward(env, agent, idx)
    """

    def __init__(
        self,
        env,
        agent,
        idx,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env: The environment to apply the wrapper
            agent: the agent whose reward will be normalized
            idx: the index of the rewards that will be normalized.
            epsilon: A stability parameter
            gamma: The discount factor that is used in the exponential moving average.
        """
        super().__init__(env)
        self.agent = agent
        self.idx = idx
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon

    def last(self, observe: bool = True):
        """Steps through the environment, normalizing the rewards returned."""
        observation, rewards, terminations, truncation, infos = self.env.last(observe)
        if self.agent != self.env.agent_selection:
            return observation, rewards, terminations, truncation, infos

        # Extracts the objective value to normalize
        to_normalize = rewards[self.idx] if isinstance(rewards, np.ndarray) else rewards  # array vs float

        self.returns = self.returns * self.gamma * (1 - terminations) + to_normalize

        # Defer normalization to gym implementation
        to_normalize = self.normalize(to_normalize)

        # Injecting the normalized objective value back into the reward vector
        # array vs float
        if isinstance(rewards, np.ndarray):
            rewards[self.idx] = to_normalize
        else:
            rewards = to_normalize

        return observation, rewards, terminations, truncation, infos

    def normalize(self, to_normalize):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return to_normalize / np.sqrt(self.return_rms.var + self.epsilon)
