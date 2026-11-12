"""Checks that both wrapper modules import and expose their wrappers.

`momaland/test/wrapper_test.py` already imports these, but `norecursedirs` keeps pytest out of the
`momaland` package, so nothing collected catches an import that has gone stale.
"""

import importlib

import pytest


@pytest.mark.parametrize("module_name", ["momaland.utils.aec_wrappers", "momaland.utils.parallel_wrappers"])
def test_wrapper_module_imports(module_name):
    """A wrapper module must import against the gymnasium version momaland requires."""
    module = importlib.import_module(module_name)

    for wrapper in ("RecordEpisodeStatistics", "LinearizeReward", "NormalizeReward"):
        assert hasattr(module, wrapper), f"{module_name} is missing {wrapper}"


def test_running_mean_std_comes_from_a_path_gymnasium_still_ships():
    """Both wrapper modules take RunningMeanStd from the same, current location."""
    from gymnasium.wrappers.utils import RunningMeanStd

    from momaland.utils import aec_wrappers, parallel_wrappers

    assert aec_wrappers.RunningMeanStd is RunningMeanStd
    assert parallel_wrappers.RunningMeanStd is RunningMeanStd
