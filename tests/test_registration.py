"""Exercise PettingZoo registration and MOMAland's vector reward factories."""

import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest
from pettingzoo import AECEnv, ParallelEnv, aec_registry, make, parallel_registry
from pettingzoo.env_registry.exceptions import NameNotFound

import momaland  # noqa: F401 - importing registers the environments
from momaland.utils.all_modules import all_environments
from momaland.utils.registration import env_id_to_name, normalize_env_id


# `all_modules` lists the legacy versioned modules. The registry must stay in sync with it: every
# module is registered for AEC, and for parallel exactly when it exposes `parallel_env`.
AEC_IDS = {normalize_env_id(name) for name in all_environments}
PARALLEL_IDS = {normalize_env_id(name) for name, module in all_environments.items() if hasattr(module, "parallel_env")}
AEC_ONLY_IDS = AEC_IDS - PARALLEL_IDS


def _momaland_ids(registry):
    return {key for key in registry if key.startswith("momaland/")}


def _resolve(entry_point):
    module_name, attr = entry_point.split(":")
    return getattr(import_module(module_name), attr)


def test_registry_coverage():
    assert _momaland_ids(aec_registry) == AEC_IDS
    assert _momaland_ids(parallel_registry) == PARALLEL_IDS


@pytest.mark.parametrize("name", sorted(all_environments))
def test_registry_matches_legacy_modules(name):
    module = all_environments[name]
    env_id = normalize_env_id(name)
    assert _resolve(aec_registry[env_id].entry_point) is module.env
    if env_id in PARALLEL_IDS:
        assert _resolve(parallel_registry[env_id].entry_point) is module.parallel_env


def test_import_does_not_load_environment_implementations():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import momaland, sys; assert not any(" "name.startswith('momaland.envs.') for name in sys.modules)",
        ],
        check=True,
    )


@pytest.mark.parametrize("env_id", sorted(AEC_IDS))
def test_aec_factory(env_id):
    env = make("aec", env_id, render_mode=None)
    try:
        assert isinstance(env, AECEnv)
        env.reset(seed=42)
        for agent in env.agent_iter(env.num_agents):
            observation, reward, terminated, truncated, info = env.last()
            assert isinstance(reward, np.ndarray)
            assert reward.shape == env.unwrapped.reward_space(agent).shape
            mask = observation.get("action_mask") if isinstance(observation, dict) else None
            action = None if terminated or truncated else env.action_space(agent).sample(mask)
            env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", sorted(PARALLEL_IDS))
def test_parallel_factory(env_id):
    env = make("parallel", env_id, render_mode=None)
    try:
        assert isinstance(env, ParallelEnv)
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, rewards, _, _, _ = env.step(actions)
        for agent, reward in rewards.items():
            assert isinstance(reward, np.ndarray)
            assert reward.shape == env.unwrapped.reward_space(agent).shape
    finally:
        env.close()


@pytest.mark.parametrize("api,constructor", [("aec", "env"), ("parallel", "parallel_env")])
def test_kwargs_and_legacy_constructor_equivalence(api, constructor):
    kwargs = {"num_timesteps": 3, "render_mode": None}
    env = make(api, "momaland/moitem_gathering-v0", **kwargs)
    module = import_module("momaland.envs.item_gathering.moitem_gathering_v0")
    legacy = getattr(module, constructor)(**kwargs)
    try:
        assert type(env) is type(legacy)
        assert env.unwrapped.num_timesteps == legacy.unwrapped.num_timesteps == 3
        assert env.unwrapped.reward_spaces == legacy.unwrapped.reward_spaces
    finally:
        env.close()
        legacy.close()


@pytest.mark.parametrize("env_id", sorted(AEC_ONLY_IDS))
def test_unsupported_parallel_factory(env_id):
    with pytest.raises(NameNotFound):
        make("parallel", env_id)


@pytest.mark.parametrize(
    "env_id",
    ["moitem_gathering_v0", "moitem_gathering-v0", "momaland/moitem_gathering_v0", "momaland/moitem_gathering-v0"],
)
def test_legacy_cli_ids(env_id):
    assert normalize_env_id(env_id) == "momaland/moitem_gathering-v0"
    assert env_id_to_name(env_id) == "moitem_gathering_v0"
    env = make("parallel", normalize_env_id(env_id), num_timesteps=1)
    try:
        env.reset(seed=42)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        assert all(terminations[agent] or truncations[agent] for agent in actions)
    finally:
        env.close()
