"""Requirement for users.

- The package `ffmpeg` is required to be installed on host system for the PNG[] -> GIF process.
"""

import sys

from pettingzoo import aec_registry, make
from PIL import Image

import momaland  # noqa: F401 - register environments
from momaland.utils.registration import env_id_to_name, normalize_env_id


def generate_gif(nameline, env_id):
    """Generates a GIF of a full environment cycle."""
    env = make("aec", normalize_env_id(env_id), render_mode="rgb_array")
    env.reset()
    imgs = []
    for _ in range(100):
        for agent in env.agent_iter(env.num_agents):  # step through every agent once with observe=True
            obs, rew, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        # save rgb_array data
        ndarray = env.render()
        im = Image.fromarray(ndarray)
        imgs.append(im)

    env.close()

    # render gif from data
    imgs[0].save(f"{nameline}.gif", save_all=True, append_images=imgs[1:], duration=40, loop=0)


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "all":
        for env_id in aec_registry:
            if env_id.startswith("momaland/"):
                generate_gif(env_id_to_name(env_id), env_id)
    else:
        generate_gif(env_id_to_name(name), name)
