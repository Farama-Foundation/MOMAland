---
title: "Parallel"
---

# Parallel
## Usage
Parallel environments can be interacted with as follows:

```python
import numpy as np
from pettingzoo import make

import momaland
from momaland.utils.parallel_wrappers import LinearizeReward

# Select the parallel API explicitly.
parallel_env = make("parallel", "momaland/momultiwalker_stability-v0", render_mode="human")

# optionally, you can scalarize the reward with weights
weights = {agent: np.array([0.7, 0.3]) for agent in parallel_env.possible_agents}
parallel_env = LinearizeReward(parallel_env, weights)

observations, infos = parallel_env.reset(seed=42)
while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}

    # vec_reward is a dict[str, numpy array]
    observations, vec_rewards, terminations, truncations, infos = parallel_env.step(actions)
parallel_env.close()
```

In `Parallel`, the returned values of `observations`, `vec_rewards`, etc. are `dict` type, where the keys are agent names, and the values are the respective data. So for `vec_rewards`, the values are `numpy` arrays.

MOMAland environments extend the base `MOParallel` class, as opposed to PettingZoo's base `Parallel` class. `MOParallel` extends `Parallel` and has a `reward_space` member.

For more detailed documentation on the Parallel API, [see the PettingZoo documentation](https://pettingzoo.farama.org/api/parallel/).

## MOParallelEnv
```{eval-rst}
.. currentmodule:: momaland.utils.env

.. autoclass:: MOParallelEnv
```
## Attributes
```{eval-rst}
.. autoattribute:: MOParallelEnv.agents

    A list of the names of all current agents, typically integers. These may be changed as an environment progresses (i.e. agents can be added or removed).

    :type: list[AgentID]

.. autoattribute:: MOParallelEnv.num_agents

    The length of the agents list.

    :type: int

.. autoattribute:: MOParallelEnv.possible_agents

    A list of all possible_agents the environment could generate. Equivalent to the list of agents in the observation and action spaces. This cannot be changed through play or resetting.

    :type: list[AgentID]

.. autoattribute:: MOParallelEnv.max_num_agents

    The length of the possible_agents list.

    :type: int

.. autoattribute:: MOParallelEnv.observation_spaces

    A dict of the observation spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gym.spaces.Space]

.. autoattribute:: MOParallelEnv.action_spaces

    A dict of the action spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gym.spaces.Space]

.. autoattribute:: MOParallelEnv.reward_spaces

    A dict of the reward spaces of every agent, keyed by name. This cannot be changed through play or resetting.

    :type: Dict[AgentID, gym.spaces.Space]
```

## Methods
```{eval-rst}
.. automethod:: MOParallelEnv.step
.. automethod:: MOParallelEnv.reset
.. automethod:: MOParallelEnv.render
.. automethod:: MOParallelEnv.close
.. automethod:: MOParallelEnv.state
.. automethod:: MOParallelEnv.observation_space
.. automethod:: MOParallelEnv.action_space
.. automethod:: MOParallelEnv.reward_space
```
