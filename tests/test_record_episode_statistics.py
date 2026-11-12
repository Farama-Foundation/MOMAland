"""Checks that the parallel RecordEpisodeStatistics wrapper survives agents finishing separately."""

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

from momaland.utils.parallel_wrappers import RecordEpisodeStatistics


class _TwoAgentEnv(ParallelEnv):
    """Two agents whose episodes end in a way the caller chooses.

    ``stagger`` finishes agent_0 a step before agent_1. ``mixed`` finishes both on the same step,
    one by termination and the other by truncation.
    """

    metadata = {"name": "record_episode_statistics_probe"}

    def __init__(self, mode):
        """Stores which ending the episode should take."""
        self.mode = mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.reward_space = {agent: Box(-1.0, 1.0, (2,)) for agent in self.possible_agents}

    def observation_space(self, agent):
        """Returns the observation space of an agent."""
        return Box(0.0, 1.0, (2,), dtype=np.float32)

    def action_space(self, agent):
        """Returns the action space of an agent."""
        return Discrete(2)

    def reset(self, seed=None, options=None):
        """Starts a new episode with both agents alive."""
        self.agents = self.possible_agents[:]
        self.timestep = 0
        observations = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}
        return observations, {agent: {} for agent in self.agents}

    def step(self, actions):
        """Advances one step and drops every agent that is done, as the parallel API requires."""
        self.timestep += 1
        observations = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}
        rewards = {agent: np.array([1.0, 1.0]) for agent in self.agents}
        if self.mode == "stagger":
            terminations = {agent: agent == "agent_0" and self.timestep == 1 for agent in self.agents}
            truncations = {agent: agent == "agent_1" and self.timestep == 2 for agent in self.agents}
        else:
            terminations = {agent: agent == "agent_0" for agent in self.agents}
            truncations = {agent: agent == "agent_1" for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos


def _play(mode):
    """Runs one episode to its end and returns the infos of the last step."""
    env = RecordEpisodeStatistics(_TwoAgentEnv(mode))
    env.reset(seed=42)
    infos = {}
    while env.agents:
        _, _, _, _, infos = env.step({agent: 0 for agent in env.agents})
    return env, infos


def test_an_agent_finishing_early_does_not_break_the_next_step():
    """The env stops reporting a finished agent, so indexing every possible agent raised KeyError."""
    env, infos = _play("stagger")

    assert "episode" in infos
    # agent_0 is paid for the one step it took, agent_1 for both of its steps.
    assert env.episode_lengths == {"agent_0": 1, "agent_1": 2}


def test_an_episode_ending_in_both_ways_at_once_is_still_recorded():
    """One agent terminating while another truncates matched neither all() check."""
    env, infos = _play("mixed")

    assert "episode" in infos
    assert infos["episode"]["l"] == {"agent_0": 1, "agent_1": 1}


@pytest.mark.parametrize("mode", ["stagger", "mixed"])
def test_reset_clears_the_counters(mode):
    """A second episode must not carry the first one's totals."""
    env, _ = _play(mode)
    env.reset(seed=42)

    assert set(env.episode_rewards) == {"agent_0", "agent_1"}
    assert all(length == 0 for length in env.episode_lengths.values())
