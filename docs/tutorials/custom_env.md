---
title: Making a Custom Environment
firstpage:
---
## Making a Custom Environment
### Setup pre-commit hooks
`pre-commit install` to initialize pre-commit hooks. `pip install pre-commit` if you don't have the package.
1. Create a new environment package in `momaland/envs/`
2. Create a new environment class in `momaland/envs/<env_name>/<env_name>.py`, this class should extend `MOParallelEnv` or `MOAECEnv`. Override the PettingZoo methods (see their [documentation](https://pettingzoo.farama.org/api/aec/)). Additionally, you should define a member `self.reward_spaces` that is a dictionary of space specifying the shape of the reward vector of each agent, as well as a method `reward_space(self, agent) -> Space` that returns the reward space of a given agent.
3. Define the factory functions to create your class: `parallel_env` returns a parallel version of the env, `env` returns an AEC version, and `raw_env` that is the pure class constructor (it is not used in practice).
- (!) Use the conversions that are defined inside our repository, e.g. `mo_parallel_to_aec` instead of `parallel_to_aec` from PZ.
4. Add a versioned constructor of your env in the directory which exports the factory functions (see `mobeach_v0.py` for an example).
5. Register the environment by adding a row to `_ENVIRONMENTS` in `momaland/envs/__init__.py`: its name, its implementation module relative to `momaland.envs`, and whether it supports the parallel API. For example:

   ```python
   ("my_env", "my_env.my_env", True),
   ```

   This registers `momaland/my_env-v0` for AEC, pointing at `momaland.envs.my_env.my_env:env`, and, when the last field is `True`, for parallel, pointing at `momaland.envs.my_env.my_env:parallel_env`.

6. Add the versioned module to `momaland/utils/all_modules.py`. `tests/test_registration.py` checks that the registry matches this list, so a missing registration or module fails the tests.
7. Run `pytest tests/test_envs.py tests/test_registration.py` to check construction, API behavior, and vector rewards.

Users can then create the environment with:

```python
from pettingzoo import make
import momaland

env = make("aec", "momaland/my_env-v0")
```

Keep MOMAland's conversion wrappers and existing constructors: the registry delegates to those factories, preserving vector rewards.
