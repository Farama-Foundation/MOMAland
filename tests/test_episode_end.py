"""Checks that environments report how an episode ended in the right `step` return value."""

import pytest

from momaland.envs.beach import mobeach_v0
from momaland.envs.gem_mining import mogem_mining_v0
from momaland.envs.item_gathering import moitem_gathering_v0
from momaland.envs.route_choice import moroute_choice_v0


def _run_parallel(env, before_step=None):
    """Steps a parallel environment with random actions until the episode ends."""
    env.reset(seed=42)
    terminations, truncations = {}, {}
    while env.agents:
        if before_step is not None:
            before_step(env)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
    env.close()
    return terminations, truncations


@pytest.mark.parametrize(
    "make_env",
    [
        moroute_choice_v0.parallel_env,
        mogem_mining_v0.parallel_env,
        lambda: mobeach_v0.parallel_env(num_timesteps=3),
    ],
)
def test_episode_end_is_a_termination(make_env):
    """The stateless bandits and Beach terminate their episodes; they are never truncated."""
    terminations, truncations = _run_parallel(make_env())

    assert all(terminations.values())
    assert not any(truncations.values())


def test_item_gathering_time_limit_truncates():
    """Running out of timesteps with items left truncates the episode."""
    terminations, truncations = _run_parallel(moitem_gathering_v0.parallel_env(num_timesteps=3))

    assert not any(terminations.values())
    assert all(truncations.values())


def test_item_gathering_all_items_gathered_terminates():
    """Gathering every item terminates the episode."""

    def remove_items(env):
        env_map = env.unwrapped.env_map
        env_map[env_map > 0] = 0

    terminations, truncations = _run_parallel(moitem_gathering_v0.parallel_env(num_timesteps=50), remove_items)

    assert all(terminations.values())
    assert not any(truncations.values())


def test_item_gathering_aec_time_limit_truncates():
    """The AEC version reports the same truncation as the parallel version."""
    env = moitem_gathering_v0.env(num_timesteps=3)
    env.reset(seed=42)
    for agent in env.agent_iter():
        _, _, termination, truncation, _ = env.last()
        if termination or truncation:
            assert not termination
            assert truncation
            env.step(None)
        else:
            env.step(env.action_space(agent).sample())
    env.close()
