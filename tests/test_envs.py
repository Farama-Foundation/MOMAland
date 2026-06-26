import os
import pickle

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, seed_test

from momaland.envs.samegame import mosame_game_v0
from momaland.test.api_test import api_test

# from momaland.test.wrapper_test import wrapper_test
from momaland.utils.all_modules import all_environments


@pytest.mark.parametrize(("name", "env_module"), list(all_environments.items()))
def test_module(name, env_module):
    _env = env_module.env(render_mode=None)
    assert str(_env) == os.path.basename(name)
    api_test(_env)
    if _env.metadata["is_parallelizable"]:
        parallel_api_test(env_module.parallel_env())
    # wrapper_test(env_module): TODO: There are some problems with the NormalizeReward wrapper.
    seed_test(env_module.env, 50)

    # TODO render_test(env_module.env)
    # TODO max_cycles_test(env_module)

    recreated_env = pickle.loads(pickle.dumps(_env))
    # TODO recreated_env.seed(42)
    api_test(recreated_env)


def test_samegame_reset_reproducible():
    env = mosame_game_v0.env(render_mode=None)
    env.reset()
    first_board, *_ = env.last()
    env.reset()
    second_board, *_ = env.last()

    assert np.array_equal(
        first_board["observation"], second_board["observation"]
    ), "no-seed reset() must reproduce the same board."
