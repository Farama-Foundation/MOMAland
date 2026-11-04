"""Implementation of multi-objective MAPPO with parameter sharing (discrete envs).

Utilizes OLS to generate weight vectors and learn a Pareto set of policies. Works for cooperative settings.
"""

import argparse
import os
import time
from functools import partial
from typing import List, NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import morl_baselines.common.weights
import numpy as np
import optax
import wandb
from distrax import MultivariateNormalDiag
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import vmap
from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from pettingzoo import make
from supersuit import agent_indicator_v0
from tqdm import tqdm

from momaland.learning.cooperative_momappo.utils import (
    policy_evaluation_mo,
    save_actor,
    strtobool,
)
from momaland.learning.utils import autotag
from momaland.utils.env import ParallelEnv
from momaland.utils.parallel_wrappers import (
    LinearizeReward,
    NormalizeReward,
    RecordEpisodeStatistics,
)
from momaland.utils.registration import env_id_to_name, normalize_env_id


def parse_args():
    """Argument parsing for the algorithm."""
    # fmt: off
    parser = argparse.ArgumentParser()

    # Env and experiment arguments
    parser.add_argument("--env-id", type=str, help="Environment ID, e.g. momaland/moitem_gathering-v0 (legacy IDs also accepted)", required=True)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="Reference point to use for the hypervolume calculation", required=True
    )
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="run in debug mode")
    parser.add_argument("--save-policies", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="save the trained policies")

    # Wandb
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="log metrics to wandb")
    parser.add_argument("--wandb-project", type=str, default="MOMAland", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="openrlbenchmark", help="the wandb's entity")
    parser.add_argument(
        "--auto-tag",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, the runs will be tagged with git tags, commit, and pull request number if possible",
    )

    # Algorithm specific arguments
    parser.add_argument("--num-weights", type=int, default=10, help="the number of different weights to train on")
    parser.add_argument("--weights-generation", type=str, default="OLS", help="The method to generate the weights - 'OLS' or 'uniform'")
    parser.add_argument("--num-steps-per-epoch", type=int, default=128, help="the number of steps per epoch (higher batch size should be better)")
    parser.add_argument("--timesteps-per-weight", type=int, default=2e3,
                        help="timesteps per weight vector")
    parser.add_argument("--update-epochs", type=int, default=2, help="the number epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=2, help="the number of minibatches (keep small in MARL)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--gae-lambda", type=float, default=0.99,
                        help="the lambda for the generalized advantage estimation")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="the epsilon for clipping in the policy objective")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="the coefficient for the entropy bonus")
    parser.add_argument("--vf-coef", type=float, default=0.8,
                        help="the coefficient for the value function loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--actor-net-arch", type=lambda x: list(map(int, x.split(','))), default=[256, 256],
                        help="actor network architecture excluding the output layer(size=action_space)")
    parser.add_argument("--critic-net-arch", type=lambda x: list(map(int, x.split(','))), default=[256, 256],
                        help="critic network architecture excluding the output layer (size=1)")
    parser.add_argument("--activation", type=str, default="tanh",
                        help="the activation function for the neural networks")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to anneal the learning rate linearly")
    parser.add_argument("--n-sample-weights", type=int, default=10,
                        help="number of weights to sample for EUM and MUL computation")

    args = parser.parse_args()
    # fmt: on
    return args


class Actor(nn.Module):
    """Actor class for the agent."""

    action_dim: Sequence[int]
    net_arch: np.ndarray
    activation: str = "tanh"

    @nn.compact
    def __call__(self, local_obs_and_id: jnp.ndarray):
        """Actor network initialization."""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.net_arch[0],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(local_obs_and_id)
        actor_mean = activation(actor_mean)
        for i in range(1, len(self.net_arch)):
            actor_mean = nn.Dense(
                self.net_arch[i],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(actor_mean)
            actor_mean = activation(actor_mean)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        probs = distrax.Categorical(logits=logits)

        return probs


class Critic(nn.Module):
    """Critic class for the agent."""

    net_arch: np.ndarray
    activation: str = "tanh"

    @nn.compact
    def __call__(self, global_obs: jnp.ndarray):
        """Actor network initialization."""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            self.net_arch[0],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(global_obs)
        critic = activation(critic)
        for i in range(1, len(self.net_arch)):
            critic = nn.Dense(
                self.net_arch[i],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """Transition info for buffer."""

    terminated: jnp.ndarray
    joint_actions: jnp.ndarray  # shape is (num_agents, action_dim)
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # shape is (num_agents, obs_dim)
    global_obs: jnp.ndarray  # shape is (global_obs_dim)
    info: jnp.ndarray


class Buffer:
    """A numpy buffer to accumulate the samples, normally faster than jax based because mutable."""

    def __init__(
        self,
        batch_size: int,
        joint_actions_shape,
        obs_shape,
        global_obs_shape,
        num_agents,
    ):
        """Buffer initialization to keep track of data between episodes."""
        self.batch_size = batch_size
        self.joint_actions = np.zeros((batch_size, *joint_actions_shape))
        self.obs = np.zeros((batch_size, *obs_shape))
        self.global_obs = np.zeros((batch_size, *global_obs_shape))
        self.value = np.zeros(batch_size)
        self.reward = np.zeros(batch_size)
        self.log_prob = np.zeros((batch_size, num_agents))
        self.terminated = np.zeros(batch_size)
        self.info = np.zeros(batch_size)  # TODO
        self.idx = 0

    def add(
        self,
        terminated: bool,
        joint_actions: np.ndarray,
        obs: np.ndarray,
        global_obs: np.ndarray,
        value: float,
        reward: float,
        log_prob: np.ndarray,
        info: dict,
    ):
        """Appending new data to the buffer."""
        self.terminated[self.idx] = terminated
        self.joint_actions[self.idx] = joint_actions
        self.obs[self.idx] = obs
        self.global_obs[self.idx] = global_obs
        self.value[self.idx] = value
        self.reward[self.idx] = reward
        self.log_prob[self.idx] = log_prob
        # TODO self.info[self.idx] = info
        self.idx += 1

    def flush(self):
        """Resetting the idx."""
        self.idx = 0

    def to_transition(self):
        """Cast to type Transition."""
        return Transition(
            terminated=jnp.array(self.terminated),
            joint_actions=jnp.array(self.joint_actions),
            obs=jnp.array(self.obs),
            global_obs=jnp.array(self.global_obs),
            value=jnp.array(self.value),
            reward=jnp.array(self.reward),
            log_prob=jnp.array(self.log_prob),
            info=jnp.array(self.info),
        )


@jax.jit
def _ma_get_pi(params, obs: jnp.ndarray):
    """Gets the actions for all agents at once. This is done with a for loop because distrax does not like vmapping."""
    return [actor.apply(params, obs[i]) for i in range(len(env.possible_agents))]


def _batched_ma_get_pi(params, obs: jnp.ndarray):
    """Gets the actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping."""
    return [actor.apply(params, obs[:, i, :]) for i in range(len(env.possible_agents))]


@jax.jit
def _ma_sample_and_log_prob_from_pi(pi: List[MultivariateNormalDiag], key: chex.PRNGKey):
    """Samples actions for all agents in all the envs at once. This is done with a for loop because distrax does not like vmapping.

    Args:
        pi (List[MultivariateNormalDiag]): List of distrax distributions for agent actions (batched over envs).
        key (chex.PRNGKey): PRNGKey to use for sampling: size should be (num_agents, 2).
    """
    return [pi[i].sample_and_log_prob(seed=key[i]) for i in range(len(env.possible_agents))]


@jax.jit
def _calculate_gae(traj_batch, last_val):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.terminated,
            transition.value,
            transition.reward,
        )
        delta = reward + args.gamma * next_value * (1 - done) - value
        gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value


def _update_minbatch(actor_critic_train_state, batch_info):
    actor_train_state, critic_train_state = actor_critic_train_state
    traj_batch, advantages, targets = batch_info

    def _loss_fn(actor_params, critic_params, traj_batch, gae, targets):
        # Batch values are in shape (batch_size, len(env.possible_agents), ...)

        # RERUN NETWORK
        pi = _batched_ma_get_pi(
            actor_params, traj_batch.obs
        )  # this is a list of distributions with batch_shape of minibatch_size and event shape of action_dim
        new_value = vmapped_get_value(critic_params, traj_batch.global_obs)
        # MA Log Prob: shape (len(env.possible_agents), minibatch_size)
        new_log_probs = jnp.array([pi[i].log_prob(traj_batch.joint_actions[:, i]) for i in range(len(env.possible_agents))])
        new_log_probs = new_log_probs.transpose()  # (minibatch_size, len(env.possible_agents))

        # Normalizes advantage (trick)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        gae = gae.reshape((-1, 1))  # (minibatch_size, 1)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (new_value - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
        value_losses = jnp.square(new_value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS FOR ALL AGENTS, AGGREGATE LOSS (sum)
        logratio = new_log_probs - traj_batch.log_prob
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()
        loss_actor1 = -ratio * gae
        loss_actor2 = -jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
        loss_per_agent = jnp.maximum(loss_actor1, loss_actor2).mean(0)  # mean across minibatch
        loss_actors = jnp.sum(loss_per_agent)  # sum across agents

        entropies = jnp.array([p.entropy().mean() for p in pi])
        entropy = entropies.mean()  # TODO check how to aggregate entropies

        total_loss = loss_actors + args.vf_coef * value_loss - args.ent_coef * entropy
        return total_loss, (value_loss, loss_actors, entropy, approx_kl)

    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
    total_loss_and_debug, grads = grad_fn(
        actor_train_state.params,
        critic_train_state.params,
        traj_batch,
        advantages,
        targets,
    )
    actor_train_state = actor_train_state.apply_gradients(grads=grads[0])
    critic_train_state = critic_train_state.apply_gradients(grads=grads[1])
    return (actor_train_state, critic_train_state), total_loss_and_debug


@jax.jit
def _update_epoch(update_state, unused):
    actor_train_state, critic_train_state, traj_batch, advantages, targets, key = update_state
    key, subkey = jax.random.split(key)
    batch_size = minibatch_size * args.num_minibatches
    permutation = jax.random.permutation(subkey, batch_size)
    batch = (traj_batch, advantages, targets)
    # flattens the num_steps_per_epoch dimensions into batch_size; keeps the other dimensions untouched (len(env.possible_agents), obs_dim, ...)
    batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[1:]), batch)
    # shuffles the full batch using permutations
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
    # Slices the shuffled batch into num_minibatches
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [args.num_minibatches, -1] + list(x.shape[1:])),
        shuffled_batch,
    )
    actor_critic_state, total_loss_and_debug = jax.lax.scan(
        _update_minbatch, (actor_train_state, critic_train_state), minibatches
    )
    update_state = (
        actor_critic_state[0],
        actor_critic_state[1],
        traj_batch,
        advantages,
        targets,
        key,
    )
    return update_state, total_loss_and_debug


def train(args, env, weights: np.ndarray, key: chex.PRNGKey):
    """MAPPO scalarizing the vector reward using weights and weighted sum."""
    num_updates = int(args.timesteps_per_weight // args.num_steps_per_epoch)
    global minibatch_size
    minibatch_size = int(args.num_steps_per_epoch // args.num_minibatches)

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
        return args.lr * frac

    env = agent_indicator_v0(env)
    for agent in env.possible_agents:
        for idx in range(env.unwrapped.reward_space(agent).shape[0]):
            env = NormalizeReward(env, agent, idx)
    _weights = {agent: weights for agent in env.possible_agents}
    env = LinearizeReward(env, _weights)  # linearizing the rewards given the weights
    env = RecordEpisodeStatistics(env)

    # Initial reset to have correct dimensions in the observations
    env.reset(seed=args.seed)

    # INIT NETWORKS
    single_obs_space = env.observation_space(env.possible_agents[0])

    key, actor_key, critic_key = jax.random.split(key, 3)
    dummy_local_obs_and_id = jnp.zeros(single_obs_space.shape)
    dummy_global_obs = jnp.zeros(env.state().shape)
    actor_params = actor.init(actor_key, dummy_local_obs_and_id)
    critic_params = critic.init(critic_key, dummy_global_obs)
    if args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        )

    actor_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=tx,
    )

    critic_train_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=tx,
    )

    # BUFFER
    buffer = Buffer(
        batch_size=args.num_steps_per_epoch,
        joint_actions_shape=(len(env.possible_agents),),
        obs_shape=(len(env.possible_agents), single_obs_space.shape[0]),
        global_obs_shape=env.state().shape,
        num_agents=len(env.possible_agents),
    )

    def _to_array_obs(obs: dict):
        """Converts a dict of observations to a numpy array of shape (num_agents, obs_dim)."""
        return np.stack([obs[agent] for agent in env.possible_agents])

    # INIT ENV
    key, subkeys = jax.random.split(key)
    obs, info = env.reset(seed=args.seed)
    episode_returns = []

    # TRAIN LOOP
    def _update_step(runner_state: Tuple[TrainState, TrainState, dict, chex.PRNGKey]):
        # COLLECT TRAJECTORIES
        def _env_step(runner_state):
            actor_state, critic_state, obs, key = runner_state

            # SELECT ACTION
            key, subkey = jax.random.split(key)
            # pi contains the normal distributions for each drone (len(env.possible_agents) x  Distribution(action_dim))
            np_obs = _to_array_obs(obs)
            pi = _ma_get_pi(actor_state.params, jnp.array(np_obs))
            action_keys = jax.random.split(subkey, len(env.possible_agents))

            # for each agent, sample an action
            actions, log_probs = zip(*_ma_sample_and_log_prob_from_pi(pi, action_keys))
            actions_dict = dict()
            for i, agent in enumerate(env.possible_agents):
                actions_dict[agent] = np.array(actions[i])
            actions = np.array(actions)
            log_probs = np.array(log_probs)

            # CRITIC STEP
            global_obs = env.state()
            value = critic.apply(critic_state.params, global_obs)

            # STEP ENV
            key, subkey = jax.random.split(key)
            obs, rewards, terminateds, truncateds, info = env.step(actions_dict)
            global current_timestep
            current_timestep += 1

            reward = np.array(list(rewards.values())).sum(axis=-1)  # team reward
            terminated = np.logical_or(
                np.any(np.array(list(terminateds.values())), axis=-1),
                np.any(np.array(list(truncateds.values())), axis=-1),
            )  # TODO handle truncations

            buffer.add(
                terminated=terminated,
                joint_actions=actions,
                obs=np_obs,
                global_obs=global_obs,
                value=value,
                reward=reward,
                log_prob=log_probs,
                info=info,
            )

            if terminated:
                team_return = sum(list(info["episode"]["r"].values()))
                if args.debug:
                    print(f"Episode return: ${team_return}, length: ${info['episode']['l']}")
                if args.track:
                    wandb.log(
                        {
                            f"charts_{weights}/episode_return": team_return,
                            f"charts_{weights}/episode_length": info["episode"]["l"][env.possible_agents[0]],
                            "global_step": current_timestep,
                        }
                    )
                obs, info = env.reset()

            runner_state = (actor_state, critic_state, obs, key)
            return runner_state

        for _ in range(args.num_steps_per_epoch):
            runner_state = _env_step(runner_state)

        # CALCULATE ADVANTAGE
        actor_train_state, critic_train_state, obs, key = runner_state
        global_obs = env.state()
        # TODO global_obs should be based on last obs, not current obs if truncated
        last_val = critic.apply(critic_train_state.params, global_obs)
        traj_batch = buffer.to_transition()

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        update_state = (
            actor_train_state,
            critic_train_state,
            traj_batch,
            advantages,
            targets,
            key,
        )
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)

        # Updates the train states (don't forget)
        actor_train_state = update_state[0]
        critic_train_state = update_state[1]
        key = update_state[-1]
        if args.track:
            wandb.log(
                {
                    f"losses_{weights}/total_loss": loss_info[0].mean(),
                    f"losses_{weights}/value_loss": loss_info[1][0].mean(),
                    f"losses_{weights}/actor_loss": loss_info[1][1].mean(),
                    f"losses_{weights}/entropy": loss_info[1][2].mean(),
                    f"losses_{weights}/approx_kl": loss_info[1][3].mean(),
                    "global_step": current_timestep,
                    "charts/SPS": current_timestep / (time.time() - start_time),
                }
            )

        buffer.flush()
        metric = traj_batch.info

        runner_state = (actor_train_state, critic_train_state, obs, key)
        return runner_state, metric

    key, subkey = jax.random.split(key)
    runner_state = (actor_train_state, critic_train_state, obs, subkey)
    for _ in tqdm(range(num_updates), desc="Updates"):
        runner_state, metric = _update_step(runner_state)

    metric = {"returned_episode_returns": np.array(episode_returns)}
    return {"runner_state": runner_state, "metrics": metric}


if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    if args.track and args.auto_tag:
        autotag()

    print("Let's go, running on", jax.devices())

    env_constructor = partial(make, "parallel", normalize_env_id(args.env_id))
    start_time = time.time()

    # Env init
    env: ParallelEnv = env_constructor()
    eval_env: ParallelEnv = env_constructor()
    eval_env = agent_indicator_v0(eval_env)

    env.reset()
    eval_env.reset()
    current_timestep = 0
    reward_dim = env.unwrapped.reward_space(env.possible_agents[0]).shape[0]

    # NN initialization and jit compiled functions
    single_action_space = env.action_space(env.possible_agents[0])
    actor = Actor(single_action_space.n, net_arch=args.actor_net_arch, activation=args.activation)
    critic = Critic(net_arch=args.critic_net_arch, activation=args.activation)
    vmapped_get_value = vmap(critic.apply, in_axes=(None, 0))
    critic.apply = jax.jit(critic.apply)

    if args.track:
        exp_name = args.exp_name
        args_dict = vars(args)
        args_dict["algo"] = exp_name
        run_name = f"{env_id_to_name(args.env_id)}__{exp_name}({args.weights_generation})__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args_dict,
            name=run_name,
            save_code=True,
        )

    ols = LinearSupport(num_objectives=reward_dim, epsilon=0.0, verbose=args.debug)
    weight_number = 1
    value = []
    if args.weights_generation == "OLS":
        w = ols.next_weight()
    elif args.weights_generation == "uniform":
        all_weights = morl_baselines.common.weights.equally_spaced_weights(reward_dim, args.num_weights)
        w = all_weights[weight_number - 1]
    else:
        raise ValueError("Weights generation method not recognized")
    while (args.weights_generation != "OLS" or not ols.ended()) and weight_number <= args.num_weights:
        out = train(args, env, w, rng)
        actor_state = out["runner_state"][0]
        _, disc_vec_return = policy_evaluation_mo(
            actor,
            actor_state,
            env=eval_env,
            num_obj=ols.num_objectives,
            gamma=args.gamma,
        )
        value.append(disc_vec_return)
        print(f"Weight {weight_number}/{args.num_weights} done!")
        print(f"Value: {disc_vec_return}, weight: {w}")
        ols.add_solution(value[-1], w)
        if args.track:
            log_all_multi_policy_metrics(
                current_front=ols.ccs,
                hv_ref_point=np.array(args.ref_point),
                reward_dim=reward_dim,
                global_step=weight_number * args.timesteps_per_weight,
                n_sample_weights=args.n_sample_weights,
            )
        if args.save_policies:
            save_actor(actor_state, w, args)
        weight_number += 1
        if args.weights_generation == "OLS":
            w = ols.next_weight()
        elif args.weights_generation == "uniform":
            w = all_weights[weight_number - 1]

    env.close()
    wandb.finish()
    print(f"total time: {time.time() - start_time}")
