"""Script to execute a policy of a trained MOMAPPO agent on a given environment."""

import argparse

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from pettingzoo import make
from supersuit import agent_indicator_v0, clip_actions_v0, normalize_obs_v0

from momaland.learning.cooperative_momappo.utils import eval_mo, load_actor_state
from momaland.utils.registration import normalize_env_id


def parse_args():
    """Parse the arguments from the command line."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--model-dir", type=str, required=True, help="the dir of the model to load.")
    parser.add_argument("--env-id", type=str, required=True, help="the id of the environment to run")
    parser.add_argument("--continuous", action="store_true", default=False, help="whether the environment is continuous or not")
    parser.add_argument("--actor-net-arch", type=lambda x: list(map(int, x.split(','))), default=[256, 256],
                        help="actor network architecture excluding the output layer(size=action_space)")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="the activation function for the neural networks")

    args = parser.parse_args()
    # fmt: on
    return args


def main():
    """Main function to execute the policy of a trained MOMAPPO agent."""
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)
    env = make("parallel", normalize_env_id(args.env_id), render_mode="human")

    # Env init
    if args.continuous:
        env = clip_actions_v0(env)
        env = normalize_obs_v0(env, env_min=-1.0, env_max=1.0)
    env = agent_indicator_v0(env)
    reward_dim = env.unwrapped.reward_space(env.possible_agents[0]).shape[0]

    # Load the actor module
    single_obs_space = env.observation_space(env.possible_agents[0])
    single_action_space = env.action_space(env.possible_agents[0])
    dummy_local_obs_and_id = jnp.zeros(single_obs_space.shape)
    env.reset(seed=args.seed)
    key, actor_key = jax.random.split(key, 2)
    if args.continuous:
        from momaland.learning.cooperative_momappo.continuous_momappo import Actor

        actor_module = Actor(
            single_action_space.shape[0],
            net_arch=args.actor_net_arch,
            activation=args.activation,
        )
        actor_state = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, dummy_local_obs_and_id),
            tx=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=0.01, eps=1e-5),  # not used
            ),
        )
    else:
        from momaland.learning.cooperative_momappo.discrete_momappo import Actor

        actor_module = Actor(
            single_action_space.shape[0],
            net_arch=args.actor_net_arch,
            activation=args.activation,
        )
        actor_state = TrainState.create(
            apply_fn=actor_module.apply,
            params=actor_module.init(actor_key, dummy_local_obs_and_id),
            tx=optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(learning_rate=0.01, eps=1e-5),  # not used
            ),
        )

    # Load the model
    actor_state = load_actor_state(args.model_dir, actor_state)
    # actor_module.apply = jax.jit(actor_module.apply)
    # Perform the replay
    vec_ret, disc_vec_return = eval_mo(actor_module=actor_module, actor_state=actor_state, env=env, num_obj=reward_dim)
    print("Done!!")
    print(vec_ret)


if __name__ == "__main__":
    main()
