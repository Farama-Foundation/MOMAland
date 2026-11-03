"""Single agent environment factory."""

from pettingzoo import make

from momaland.envs.item_gathering.map_utils import DEFAULT_MAP, generate_map
from momaland.utils.parallel_wrappers import CentraliseAgent


def get_map_4_O():
    """Generate a map with 4 objectives."""
    return generate_map(rows=8, columns=8, item_distribution=(3, 4, 2, 1), num_agents=2, seed=1)


def get_map_2_O():
    """Generate a map with 2 objectives."""
    return generate_map(rows=8, columns=8, item_distribution=(4, 6), num_agents=2, seed=1)


def make_single_agent_ig_env(objectives=3):
    """Create a centralised agent environment for the Item Gathering domain."""
    if objectives == 2:
        env_map = get_map_2_O()
    elif objectives == 4:
        env_map = get_map_4_O()
    else:
        env_map = DEFAULT_MAP
    ig_env = make(
        "parallel", "momaland/moitem_gathering-v0", initial_map=env_map, num_timesteps=50, randomise=False, render_mode=None
    )
    return CentraliseAgent(ig_env, action_mapping=True)


def make_single_agent_bpd_env(size="small"):
    """Create a centralised agent environment for the MO Beach Problem domain."""
    if size == "small":
        bpd_env = make(
            "parallel",
            "momaland/mobeach-v0",
            num_timesteps=5,
            num_agents=10,
            reward_mode="team",
            sections=3,
            capacity=2,
            type_distribution=(0.7, 0.3),
            position_distribution=(0.5, 0.0, 0.5),
        )
    else:
        bpd_env = make(
            "parallel",
            "momaland/moitem_gathering-v0",
            num_timesteps=1,
            num_agents=50,
            reward_mode="team",
            sections=5,
            capacity=3,
            type_distribution=(0.7, 0.3),
            position_distribution=(0.0, 0.5, 0.0, 0.5, 0.0),
        )
    return CentraliseAgent(bpd_env, action_mapping=True, reward_type="average")


def make_single_agent_mw_env(reward_type="average", n_walkers=3):
    """Create a centralised agent environment for the Multiwalker domain."""
    mw_env = make("parallel", "momaland/momultiwalker_stability-v0", n_walkers=n_walkers, remove_on_fall=False)
    return CentraliseAgent(mw_env, action_mapping=False, reward_type=reward_type)
