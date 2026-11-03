"""Register MOMAland environments with PettingZoo without importing their implementations."""

from pettingzoo import register


# (name, implementation module, supports parallel execution)
_ENVIRONMENTS = (
    ("mobeach", "beach.beach", True),
    ("momultiwalker_stability", "multiwalker_stability.momultiwalker_stability", True),
    ("catch", "crazyrl.catch.catch", True),
    ("surround", "crazyrl.surround.surround", True),
    ("escort", "crazyrl.escort.escort", True),
    ("moitem_gathering", "item_gathering.item_gathering", True),
    ("moingenious", "ingenious.ingenious", False),
    ("mopistonball", "pistonball.pistonball", True),
    ("moroute_choice", "route_choice.route_choice", True),
    ("moconnect4", "connect4.connect4", False),
    ("mobreakthrough", "breakthrough.breakthrough", False),
    ("mosame_game", "samegame.same_game", False),
    ("mogem_mining", "gem_mining.gem_mining", True),
)

for _name, _module, _parallel in _ENVIRONMENTS:
    register("aec", f"momaland/{_name}-v0", entry_point=f"momaland.envs.{_module}:env")
    if _parallel:
        register("parallel", f"momaland/{_name}-v0", entry_point=f"momaland.envs.{_module}:parallel_env")
