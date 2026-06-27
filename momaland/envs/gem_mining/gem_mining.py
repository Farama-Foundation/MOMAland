"""Gem Mining Problem.

Combining the Mining Day environment from:
    Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning, PhD Thesis, University of Amsterdam, 2016. (Which contains the multi-objective planning version.)
And the Gem Mining problem from:
    Eugenio Bargiacchi, Timothy Verstraeten, Diederik M. Roijers, Ann Nowé and Hado van Hasselt - Learning to Coordinate with Coordination Graphs in Repeated Single-Stage Multi-Agent Decision Problems. In ICML 2018, Stockholm, July 2018. (Which contains the single-objective learning version.)

Created on Tue Oct 24 16:31:14 2023

@author: dmroijers
"""

import functools
import random
from typing_extensions import override

import numpy as np
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers

from momaland.utils.conversions import mo_parallel_to_aec
from momaland.utils.env import MOParallelEnv


def parallel_env(**kwargs):
    """Parallel env factory function for the gem mining domain."""
    return raw_env(**kwargs)


def env(**kwargs):
    """Auto-wrapper for the MO Gem Mining problem.

    Args:
        **kwargs: keyword args to forward to the parallel_env function

    Returns:
        A fully wrapped AEC env
    """
    env = parallel_env(**kwargs)
    env = mo_parallel_to_aec(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the gem mining domain."""
    return MOGemMining(**kwargs)


class MOGemMining(MOParallelEnv, EzPickle):
    """Environment for MO-GemMining domain.

    ## Observation Space
    The observation space is a cBox of the number of agents in length.
    As this is a stateless environment, all agents receive a "0" observation each timestep.

    ## Action Space
    The action space is discrete set of integers for each agent, and is agent-specific.
    Each integer represents the ID of a mine (i.e., local reward function) which is reachable from the village (i.e., agent).
    Selecting an action represents sending the workers that live in a given village to the corresponding mine.

    ## Reward Space
    The reward space is a vector containing rewards in each objective (customizable).
    Each objective corresponds to a type of gem that can be found at the mines.
    The rewards correspond to the total number of gems of each type found at all the mines together at a given timestep.
    Please note that as this is a fully cooperative environment all agents receive the same reward vectors.

    ## Starting State
    As this is a state-less environment the "state" is just a default value. (See Observation Space.)

    ## Episode Termination
    As this is a state-less environment there isn't really an episode.
    Hence the episode terminates after each timestep.

    ## Episode Truncation
    Each "episode" last 1 timestep (due to the bandit setting).

    ## Arguments
    - `num_agents: number of agents (i.e., villages) in the Gem Mining instance
    - num_objectives: number of objectives (i.e., gem types), each mine has a probability of generating gems of any type at any timesteps
    - min_connectivity: the minimum number of mines each agent is connected to. Should be greater or equal to 2
    - max_connectivity: the maximum number of mines each agent is connected to. Should be greater or equal to min_connectivity
    - min_workers: the minimum number of workers per village (agent). Should be greater or equal to 1.
    - max_workers: the maximum number of workers per village (agent). Should be greater or equal to min_workers.
    - min_prob: the minimum (Bernoulli) probability of finding a gem (per type) at a mine, excluding worker bonus
    - max_prob: the maximum (Bernoulli) probability of finding a gem (per type) at a mine, excluding worker bonus
    - trunc_probability: upper limit to the probability of finding a gem after adding the worker bonus
    - w_bonus: worker bonus; the probability of finding a gem is multiplied by w_bonus^(w-1), where w is the number of workers at a mine
    - correlated_objectives: if true, the probability of mining a given type of gem at a mine is negatively correlated to finding a gem of another type, and the (non-bonus) expectation of finding any gem is at most max_prob per mine per timestep.
    - num_timesteps: number of timesteps (stateless, therefore defaultly set to 1 timestep)
    - render_mode: render mode
    - seed: This environment is generated randomly using the provided seed. Defaults to 42.

    ## Credits
    The code was based on previous code by Diederik Roijers and Eugenio Bargiacchi (in different programming languages), and reimplemented.
    """

    metadata = {"render_modes": ["human"], "name": "mogem_mining_v0"}

    def __init__(
        self,
        num_agents=20,
        num_objectives=2,
        min_connectivity=2,
        max_connectivity=4,
        min_workers=1,
        max_workers=5,
        min_prob=0.01,
        max_prob=0.50,
        trunc_probability=0.9,
        w_bonus=1.03,
        correlated_objectives=True,
        num_timesteps=1,
        render_mode=None,
        seed=42,
    ):
        """Initializes the gem mining environment.

        Args:
            num_agents: number of agents (i.e., villages) in the Gem Mining instance
            num_objectives: number of objectives (i.e., gem types), each mine has a probability of generating gems of any type at any timesteps
            min_connectivity: the minimum number of mines each agent is connected to. Should be greater or equal to 2
            max_connectivity: the maximum number of mines each agent is connected to. Should be greater or equal to min_connectivity
            min_workers: the minimum number of workers per village (agent). Should be greater or equal to 1.
            max_workers: the maximum number of workers per village (agent). Should be greater or equal to min_workers.
            min_prob: the minimum (Bernoulli) probability of finding a gem (per type) at a mine, excluding worker bonus
            max_prob: the maximum (Bernoulli) probability of finding a gem (per type) at a mine, excluding worker bonus
            trunc_probability: upper limit to the probability of finding a gem after adding the worker bonus
            w_bonus: worker bonus; the probability of finding a gem is multiplied by w_bonus^(w-1), where w is the number of workers at a mine
            correlated_objectives: if true, the probability of mining a given type of gem at a mine is negatively correlated to finding a gem of another type, and the (non-bonus) expectation of finding any gem is at most max_prob per mine per timestep.
            num_timesteps: number of timesteps (stateless, therefore always 1 timestep)
            render_mode: render mode
            seed: This environment is generated randomly using the provided seed. Defaults to 42.
        """
        EzPickle.__init__(
            self,
            num_agents,
            num_objectives,
            min_connectivity,
            max_connectivity,
            min_workers,
            max_workers,
            min_prob,
            max_prob,
            trunc_probability,
            w_bonus,
            correlated_objectives,
            num_timesteps,
            render_mode,
            seed,
        )
        self.num_timesteps = num_timesteps
        self.episode_num = 0
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.num_mines = num_agents + max_connectivity - 1

        self.num_objectives = num_objectives
        self.worker_bonus = w_bonus
        self.truncation_probability = trunc_probability

        self.random = random.Random(seed)
        self.np_random = np.random.default_rng(seed)

        # determine the number of workers per village (agent):
        lst = list(range(min_workers, max_workers + 1))
        self.workers = {agent: self.random.choice(lst) for agent in self.agents}

        # determine the base probabilities of finding a gem per type per mine
        self.base_probabilities = dict()
        for i in range(self.num_mines):
            self.base_probabilities[i] = np.zeros(self.num_objectives, dtype=int)
            left = max_prob - self.num_objectives * min_prob
            for j in range(self.num_objectives):
                if correlated_objectives:
                    pj = self.random.uniform(min_prob, left)
                    left = left - pj
                else:
                    pj = self.random.uniform(min_prob, max_prob)
                self.base_probabilities[i][j] = pj
            self.random.shuffle(self.base_probabilities[i])

        # action spaces are numbers (IDs) of the mines where each agent can go
        self.action_spaces = dict()
        for i in range(num_agents):
            connect = self.random.randint(min_connectivity, max_connectivity)
            self.action_spaces[f"agent_{i}"] = Discrete(connect, start=i)
        # stateless setting, agents receive a constant '0' as an observation in each timestep
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(
                        low=0,
                        high=self.num_agents,
                        shape=(1,),
                        dtype=np.float32,
                    )
                ]
                * num_agents,
            )
        )
        self.reward_spaces = dict(zip(self.agents, [Box(low=0, high=self.num_mines, shape=(num_objectives,))] * num_agents))

        # set truncations to false (no agents are ever lost)
        self.truncations = {agent: False for agent in self.agents}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        return self.reward_spaces[agent]

    @override
    def render(self):
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        observations = {agent: np.array([0], dtype=np.float32) for i, agent in enumerate(self.agents)}
        self.episode_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """Steps in the environment.

        Args:
            actions: a dict of actions, keyed by agent names

        Returns: a tuple containing the following items in order:
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # - Observations -#
        # return constant observations '0' as this is a stateless setting
        observations = {agent: np.array([0], dtype=np.float32) for agent in self.agents}

        # - Rewards -#
        # First, calculate the number of workers ending up at each mine:
        workers_at_mine = np.zeros(self.num_mines, dtype=np.int32)
        for agent, mine in actions.items():
            workers_at_mine[mine] = workers_at_mine[mine] + self.workers[agent]
        # The rewards are based on Bernoulli (binomial(1)) experiments per mine per objective
        reward_vec = np.zeros(self.num_objectives, dtype=np.float32)
        for i in range(self.num_mines):
            if workers_at_mine[i] > 0:
                bonus = pow(self.worker_bonus, workers_at_mine[i])
                for j in range(self.num_objectives):
                    prob = bonus * self.base_probabilities[i][j]
                    if prob > self.truncation_probability:
                        prob = self.truncation_probability
                    outcome = self.np_random.binomial(size=1, n=1, p=prob)
                    reward_vec[j] = reward_vec[j] + outcome[0]
        # every agent gets the same reward vector (fully cooperative)
        rewards = {agent: reward_vec for agent in self.agents}

        # - Infos -#
        # typically there won't be any information in the infos, but there must still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        # stateless bandit setting where each episode only lasts 1 timestep
        self.terminations = {agent: True for agent in self.agents}
        self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos

    # - Helper Methods (prefix by _) -#
    def _random_action(self):
        """Draw a valid random action for the MOGemMining instance.

        Returns:
            A valid random joint action
        """
        return {i: self.action_space(i).sample() for i in self.agents}
