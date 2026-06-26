"""Item Gathering environment, based on the environment from the paper below.

Johan Källström and Fredrik Heintz. 2019. Tunable Dynamics in Agent-Based Simulation using Multi-Objective
Reinforcement Learning. Presented at the Adaptive and Learning Agents Workshop at AAMAS 2019.
https://liu.diva-portal.org/smash/record.jsf?pid=diva2%3A1362933&dswid=9018

Notes:
    - In contrast to the original environment, the observation space is a 2D array of integers, i.e.,
    the map of the environment, with 0 for empty cells, negative integers for agents, positive integers for items.
    - The number of agents and items is configurable, by providing an initial map.
    - If no initial map is provided, the environment uses a default map

Central observation:
    - If the central_observation flag is set to True, then the environment includes in the implementation:
        - a central observation space: self.central_observation_space
        - a central observation function: self.state()
    The central_observation flag and the associated methods described above are used by the CentralisedAgent wrapper
"""

import random
import warnings
from copy import deepcopy
from os import path
from typing_extensions import override

import numpy as np
import pygame
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete, Tuple
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers

from momaland.envs.item_gathering.asset_utils import get_colored
from momaland.envs.item_gathering.map_utils import DEFAULT_MAP, randomise_map
from momaland.utils.conversions import mo_parallel_to_aec
from momaland.utils.env import MOParallelEnv


ACTIONS = {
    0: np.array([0, 0], dtype=np.int32),  # stay
    1: np.array([-1, 0], dtype=np.int32),  # up
    2: np.array([1, 0], dtype=np.int32),  # down
    3: np.array([0, -1], dtype=np.int32),  # left
    4: np.array([0, 1], dtype=np.int32),  # right
}


def parallel_env(**kwargs):
    """Parallel env factory function for the item gathering problem."""
    return raw_env(**kwargs)


def env(**kwargs):
    """Autowrapper for the item gathering problem.

    Args:
        **kwargs: keyword args to forward to the parallel_env function

    Returns:
        A fully wrapped env
    """
    env = raw_env(**kwargs)
    env = mo_parallel_to_aec(env)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the item gathering problem."""
    return MOItemGathering(**kwargs)


class MOItemGathering(MOParallelEnv, EzPickle):
    """A `Parallel` multi-objective environment of the Item Gathering problem.

    ## Observation Space
    The observation space is a tuple containing the agent id (a negative integer) and the 2D map observation, where
    0 is an empty cell, negative integers represent agent IDs, and positive integers represent items

    ## Action Space
    The action space is a Discrete space, where:
    -   0: stay
    -   1: up
    -   2: down
    -   3: left
    -   4: right

    ## Reward Space
    The reward space is a vector containing rewards for each type of items available in the environment

    ## Starting State
    The initial position of the agent is determined by the 1 entries of the initial map.

    ## Episode Termination
    The episode is terminated if all the items have been gathered.

    ## Episode Truncation
    The episode termination occurs if the maximum number of timesteps is reached.

    ## Arguments
    - 'num_timesteps': number of timesteps to run the environment for. Default: 10
    - 'initial_map': map of the environment. Default: 8x8 grid, 2 agents, 3 objectives (Källström and Heintz, 2019)
    - 'randomise': whether to randomise the map, at each episode. Default: False
    - 'reward_mode': reward mode for the environment ('individual' or 'team'). Default: 'individual'
    - 'pickup_probabilities': the probabilities of picking up each item type, by default set to None which means all items are picked up with probability 1.0.
    - 'render_mode': render mode for the environment. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "moitem_gathering_v0",
        "is_parallelizable": True,
        "central_observation": True,
        "render_fps": 30,
    }

    def __init__(
        self,
        num_timesteps=10,
        initial_map=DEFAULT_MAP,
        randomise=False,
        reward_mode="individual",
        pickup_probabilities=None,
        render_mode=None,
    ):
        """Initializes the item gathering domain.

        Args:
            num_timesteps: number of timesteps to run the environment for
            initial_map: map of the environment
            randomise: whether to randomise the map, at each episode
            reward_mode: reward mode for the environment, 'individual' or 'team'. Default: 'individual'
            pickup_probabilities: the probabilities of picking up each item type, by default set to None which means all items are picked up with probability 1.0.
            render_mode: render mode for the environment
        """
        EzPickle.__init__(
            self,
            num_timesteps,
            initial_map,
            randomise,
            reward_mode,
            pickup_probabilities,
            render_mode,
        )
        self.num_timesteps = num_timesteps
        self.current_timestep = 0
        self.render_mode = render_mode
        self.randomise = randomise
        if reward_mode not in ["individual", "team"]:
            self.reward_mode = "individual"
            warnings.warn("reward_mode must be either 'individual' or 'team', defaulting to 'individual'.")
        else:
            self.reward_mode = reward_mode

        # check if the initial map has any entries equal to 1
        assert len(np.argwhere(initial_map == 1).flatten()) > 0, "The initial map does not contain any agents (1s)."
        self.initial_map = initial_map

        # self.env_map is the working copy used in each episode. self.initial_map should not be modified.
        self.agent_positions, self.env_map = self._init_map_and_positions(deepcopy(self.initial_map))

        self.possible_agents = ["agent_" + str(r) for r in range(len(self.agent_positions))]
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.time_num = 0

        # determine the number of item types and maximum number of each item
        all_map_entries = np.unique(self.env_map, return_counts=True)
        self.item_dict = {}
        # create an item dictionary, with item value in map as key and index as value
        for i, item in enumerate(all_map_entries[0][np.where(all_map_entries[0] > 0)]):
            self.item_dict[item] = i

        indices_of_items = np.argwhere(all_map_entries[0] > 0).flatten()
        item_counts = np.take(all_map_entries[1], indices_of_items)
        self.num_objectives = len(item_counts)

        # initialize stochastic behavior if selected
        self.rng = np.random.default_rng()
        self.pickup_probs = pickup_probabilities if pickup_probabilities is not None else np.ones(self.num_objectives)

        assert len(item_counts) > 0, "There are no resources in the map."

        self.action_spaces = dict(zip(self.agents, [Discrete(len(ACTIONS))] * len(self.agent_positions)))

        # observations are a tuple of agent id (in the map) and the map observation
        # the map observation is a 2D array of integers, where each integer represents either agents or items.
        # 0 for empty, negative integers for agents, positive integers for items
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Tuple(
                        (
                            Discrete(len(self.possible_agents), start=-(len(self.possible_agents))),
                            Box(
                                low=-(len(self.possible_agents)),
                                high=self.num_objectives,
                                shape=self.env_map.shape,
                                dtype=np.int64,
                            ),
                        )
                    )
                ]
                * len(self.agent_positions),
            )
        )

        self.central_observation_space = Box(
            low=-(len(self.possible_agents)),
            high=self.num_objectives,
            shape=self.env_map.flatten().shape,
            dtype=np.int64,
        )

        self.reward_spaces = dict(
            zip(self.agents, [Box(low=0, high=max(item_counts), shape=(self.num_objectives,))] * len(self.agent_positions))
        )

        # pygame
        self.size = 5
        self.cell_size = (64, 64)
        self.window_size = (
            self.env_map.shape[1] * self.cell_size[1],
            self.env_map.shape[0] * self.cell_size[0],
        )
        self.clock = None
        self.agent_imgs = []
        self.item_imgs = []
        self.map_bg_imgs = []
        self.window = None

    @override
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasiuspspom.farama.org/api/spaces/
        return self.observation_spaces[agent]

    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        return self.reward_spaces[agent]

    def get_central_observation_space(self):
        """Returns the central observation space."""
        return self.central_observation_space

    @override
    def render(self):
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return
        if self.window is None:
            # pygame.init() also starts the unused audio mixer and joystick subsystems, whose device
            # enumeration adds ~0.4s of startup (see Farama-Foundation/MOMAland#71). Only the display
            # is needed, and it is initialized on the human render path below.
            get_colored("item", len(self.item_dict))
            get_colored("agent", len(self.agents))

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Item Gathering")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            if not self.agent_imgs:
                agents = [path.join(path.dirname(__file__), "assets/agent.png")]
                for i in range(len(self.agents) - 1):
                    agents.append(path.join(path.dirname(__file__), f"assets/colored/agent{i}.png"))
                self.agent_imgs = [pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in agents]
            if not self.map_bg_imgs:
                map_bg_imgs = [
                    path.join(path.dirname(__file__), "assets/map_bg1.png"),
                    path.join(path.dirname(__file__), "assets/map_bg2.png"),
                ]
                self.map_bg_imgs = [
                    pygame.transform.scale(pygame.image.load(f_name), self.cell_size) for f_name in map_bg_imgs
                ]
            if not self.item_imgs:
                items = [path.join(path.dirname(__file__), "assets/item.png")]
                for i in range(len(self.item_dict) - 1):
                    items.append(path.join(path.dirname(__file__), f"assets/colored/item{i}.png"))
                self.item_imgs = [
                    pygame.transform.scale(pygame.image.load(f_name), (0.6 * self.cell_size[0], 0.6 * self.cell_size[1]))
                    for f_name in items
                ]

        for i in range(self.env_map.shape[0]):
            for j in range(self.env_map.shape[1]):
                # background
                check_board_mask = i % 2 ^ j % 2
                self.window.blit(
                    self.map_bg_imgs[check_board_mask],
                    np.array([j, i]) * self.cell_size[0],
                )

                # agents
                if [i, j] in self.agent_positions.tolist():
                    ind = self.agent_positions.tolist().index([i, j])
                    self.window.blit(self.agent_imgs[ind], tuple(np.array([j, i]) * self.cell_size[0]))

                # items
                elif int(self.env_map[i, j]) > 0:
                    item_value = int(self.env_map[i, j])
                    if item_value in self.item_dict:
                        ind = self.item_dict[item_value]
                        self.window.blit(self.item_imgs[ind], tuple(np.array([j + 0.22, i + 0.25]) * self.cell_size[0]))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    @override
    def close(self):
        pass
        # This breaks the pickle tests
        # if self.render_mode is not None:
        #     del_colored()

    @override
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Reset the environment map
        if self.randomise:
            self.env_map = randomise_map(deepcopy(self.initial_map))
        else:
            self.env_map = deepcopy(self.initial_map)

        self.agent_positions, self.env_map = self._init_map_and_positions(self.env_map)

        map_obs = self.state()
        # observations are a tuple of agent id (in the map) and the map observation
        observations = {agent: (-(i + 1), map_obs) for i, agent in enumerate(self.agents)}
        self.time_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    @override
    def _is_valid_position(self, position):
        row_valid = 0 <= position[0] < len(self.env_map)
        col_valid = 0 <= position[1] < len(self.env_map[0])
        return row_valid and col_valid

    def step(self, actions):
        """Steps in the environment.

        Args:
            actions: a dict of actions, keyed by agent names

        Returns: a tuple containing the following items in order:
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        new_positions = deepcopy(self.agent_positions)
        # Apply actions and update system state
        for i, agent in enumerate(self.agents):
            act = actions.get(agent)  # use .get() for safety
            if act is None:
                continue
            new_position = self.agent_positions[i] + ACTIONS[act]
            if self._is_valid_position(new_position):
                # update the position, if it is a valid step
                new_positions[i] = new_position

        # Check for collisions, resolve here only the collision, have final new position list at end
        collisions = []
        # initial collision check, verify all agent pairs
        new_positions, collisions = self._verify_collisions(collisions, new_positions)

        max_collisions = len(self.possible_agents) * 50
        while len(collisions) > 0 and max_collisions > 0:
            new_positions, collisions = self._verify_collisions(collisions, new_positions)
            max_collisions -= 1
        assert collisions == [], "Collision resolution failed"

        # update the agent positions now that all collisions are resolved
        self.agent_positions = deepcopy(new_positions)

        # initialise rewards and observations
        rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}

        # --- Stochastic Item Pickup ---
        for i in range(len(self.agent_positions)):
            agent_id = self.agents[i]
            agent_pos = self.agent_positions[i]
            value_in_cell = self.env_map[agent_pos[0], agent_pos[1]]

            if value_in_cell > 0:
                item_idx = self.item_dict[value_in_cell]
                prob = self.pickup_probs[item_idx]

                # Attempt to pick up the item based on its probability
                if random.random() < prob:
                    # Success: grant reward and remove item
                    rewards[agent_id][item_idx] += 1
                    self.env_map[agent_pos[0], agent_pos[1]] = 0
                # else: Failure: do nothing. The agent stays on the square,
                # the item remains, no reward is given.

        # if reward mode is teams, sum the rewards for all agents
        if self.reward_mode == "team":
            # Ensure all agents get the same summed reward vector
            total_rewards = np.sum(list(rewards.values()), axis=0)
            rewards = {agent: total_rewards for agent in self.agents}

        map_obs = self.state()
        observations = {agent: (-(i + 1), map_obs) for i, agent in enumerate(self.agents)}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        # termination occurs when all items or gathered
        # truncation occurs if all timesteps are exhausted
        self.time_num += 1
        env_termination = bool(np.sum(self.env_map[self.env_map > 0]) == 0)  # Check only items
        env_truncation = bool(self.time_num >= self.num_timesteps)
        self.terminations = {agent: env_termination for agent in self.agents}
        self.truncations = {agent: env_truncation for agent in self.agents}
        if env_termination or env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos

    def state(self):
        """Returns the global observation of the map passed to each agent at the end of a timestep.

        Returns: a 2D Numpy array with the following items encoded:
        - 0 is empty space
        - -1, -2, -3 ... denote the positions of the agents with the specified -(agent_id+1)
        - 1, 2, 3 ... denote the locations of items representing different objectives
        """
        map_obs = deepcopy(self.env_map)
        # encode the agent positions with the negative agent id, starting from -1
        for i in range(len(self.agent_positions)):
            map_obs[self.agent_positions[i][0], self.agent_positions[i][1]] = -(i + 1)
        return map_obs

    def _verify_collisions(self, check_set, positions):
        """Function to check for collisions between agents.

        Args:
            check_set: the set of positions to check for collisions
            new_positions: the new positions of the agents

        Returns:
            new_positions: the new positions of the agents after resolving collisions
            collisions: the list of agents that collided and for which the position changed
        """
        collisions = []
        if len(check_set) == 0:
            for i in range(len(positions) - 1):
                for j in range(i + 1, len(positions)):
                    positions, collisions = self._verify_positions(i, j, positions, collisions)
        else:
            for i in check_set:
                for j in range(len(positions)):
                    if i != j:
                        positions, collisions = self._verify_positions(i, j, positions, collisions)
        return positions, collisions

    def _verify_positions(self, i, j, positions, collisions):
        """Function to check for collisions between agents and re-assign them the old position, if needed."""
        # if agents are on the same location
        if np.array_equal(positions[i], positions[j]):
            # randomly choose between colliding agents
            choice = random.choice([i, j])
            # and re-assign the position of the selected agent to its old position
            positions[choice] = self.agent_positions[choice]
            collisions.append(choice)
        return positions, collisions

    def _init_map_and_positions(self, env_map):
        """Initialise the map and agent positions."""
        agent_positions = np.argwhere(env_map == 1)  # store agent positions in separate list
        env_map[env_map == 1] = 0  # remove agent starting positions from map

        # This can fail if there are no items, guard it
        if np.any(env_map > 0):
            # shift the item indices to start from 1
            env_map[env_map > 0] -= min(env_map[env_map > 0]) - 1
        return agent_positions, env_map
