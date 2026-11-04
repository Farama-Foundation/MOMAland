from pettingzoo import make

import momaland  # noqa: F401 - register environments


env = make("parallel", "momaland/mopistonball-v0", render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}

    observations, vec_rewards, terminations, truncations, infos = env.step(actions)
env.close()
