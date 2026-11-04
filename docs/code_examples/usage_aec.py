from pettingzoo import make

import momaland  # noqa: F401 - register environments


env = make("aec", "momaland/mopistonball-v0", render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, vec_reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
