"""Beach problem domain.

From Mannion, P., Devlin, S., Duggan, J., and Howley, E. (2018). Reward shaping for knowledge-based multi-objective multi-agent reinforcement learning.
"""

import colorsys
import functools
import random
import warnings
from typing_extensions import override

import numpy as np
import pygame
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers

from momaland.utils.conversions import mo_parallel_to_aec
from momaland.utils.env import MOParallelEnv


LEFT = -1
RIGHT = 1
STAY = 0
MOVES = [LEFT, STAY, RIGHT]
NUM_OBJECTIVES = 2


def parallel_env(**kwargs):
    """Parallel env factory function for the beach problem domain."""
    return raw_env(**kwargs)


def env(**kwargs):
    """Autowrapper for the beach domain.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped env
    """
    env = parallel_env(**kwargs)
    env = mo_parallel_to_aec(env)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the beach problem domain."""
    return MOBeachDomain(**kwargs)


class MOBeachDomain(MOParallelEnv, EzPickle):
    """A `Parallel` 2-objective environment of the Beach problem domain.

    ## Observation Space
    The observation space is a continuous box with the length `5` containing:
     - agent type
     - section id (where the agent is)
     - section capacity
     - section consumption
     - percentage of agents of the agent's type in the section

    Example:
    `[a_type, section_id, section_capacity, section_consumption, %_of_a_of_current_type]`

    ## Action Space
    The action space is a Discrete space [0, 1, 2], corresponding to moving left, moving right, staying in place.

    ## Reward Space
    The reward space is a 2D vector containing rewards for two different modes ('individual' or 'team') for:
    - the occupation level
    - the mixture level
    If the mode is 'individual', the reward is given for the currently occupied section.
    If the mode is 'team', the reward is summed over all sections.

    ## Starting State
    The initial position is a uniform random distribution of agents over the sections. This can be changed via the
    'position_distribution' argument. The agent types are also randomly distributed according to the
    'type_distribution' argument. The default is a uniform distribution over all types.

    ## Episode Termination
    The episode is terminated if num_timesteps is reached. The default value is 100.
    Agents only receive the reward after the last timestep.

    ## Episode Truncation
    The problem is not truncated. It has a maximum number of timesteps.

    ## Arguments
    - 'num_timesteps (int)': number of timesteps in the domain. Default: 1
    - 'num_agents (int)': number of agents in the domain. Default: 100
    - 'reward_mode (str)': the reward mode to use ('individual', or 'team'). Default: individual
    - 'sections (int)': number of beach sections in the domain. Default: 6
    - 'capacity (int)': capacity of each beach section. Default: 7
    - 'type_distribution (tuple)': the distribution of agent types in the domain. Default: 2 types equally distributed (0.3, 0.7).
    - 'position_distribution (tuple)': the initial distribution of agents in the domain. Default: uniform over all sections (None).
    - 'render_mode (str)': render mode. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "mobeach_v0",
        "central_observation": True,
        "render_fps": 5,
    }

    def __init__(
        self,
        num_timesteps=1,
        num_agents=100,
        reward_mode="individual",
        sections=6,
        capacity=7,
        type_distribution=(0.3, 0.7),
        position_distribution=None,
        render_mode=None,
    ):
        """Initializes the beach domain.

        Args:
            sections: number of beach sections in the domain
            capacity: capacity of each beach section
            num_agents: number of agents in the domain
            reward_mode: the reward mode to use ('individual', or 'team'). Default: individual
            type_distribution: the distribution of agent types in the domain. Default: 2 types equally distributed.
            position_distribution: the initial distribution of agents in the domain. Default: uniform over all sections.
            num_timesteps: number of timesteps in the domain
            render_mode: render mode
        """
        EzPickle.__init__(
            self,
            num_timesteps,
            num_agents,
            reward_mode,
            sections,
            capacity,
            type_distribution,
            position_distribution,
            render_mode,
        )
        if reward_mode not in ["individual", "team"]:
            self.reward_mode = "individual"
            warnings.warn("Invalid reward_mode. Must be either 'individual' or 'team'. Defaulting to 'individual'.")
        else:
            self.reward_mode = reward_mode
        self.sections = sections
        self.resource_capacities = [capacity for _ in range(sections)]
        self.num_timesteps = num_timesteps
        self.episode_num = 0
        self.type_distribution = type_distribution
        if position_distribution is None:
            self.position_distribution = [1 / sections for _ in range(sections)]
        else:
            assert (
                len(position_distribution) == self.sections
            ), "number of sections should be equal to the length of the provided position_distribution:"
            self.position_distribution = position_distribution

        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self._types, self._state = self._init_state()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        self.action_spaces = dict(zip(self.agents, [Discrete(len(MOVES))] * num_agents))
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Box(
                        low=0,
                        high=self.num_agents,
                        # Observation form:
                        # agent type, section id, section capacity, section consumption, % of agents of current type
                        shape=(5,),
                        dtype=np.float32,
                    )
                ]
                * num_agents,
            )
        )

        self.central_observation_space = Box(
            low=0,
            high=self.num_agents,
            # Observation form:
            # agents * [agent type, section id, section capacity, section consumption, % of agents of current type]
            shape=(self.num_agents * 5,),
            dtype=np.float32,
        )

        # maximum capacity reward can be calculated  by calling the _global_capacity_reward()
        optimal_consumption = [capacity for _ in range(sections)]
        optimal_consumption[-1] = max(self.num_agents - ((sections - 1) * capacity), 0)
        max_r = _global_capacity_reward(self.resource_capacities, optimal_consumption)
        self.reward_spaces = dict(
            zip(
                self.agents,
                [Box(low=0, high=max_r, shape=(NUM_OBJECTIVES,))] * num_agents,
            )
        )

        # pygame rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.section_width = 150
        self.window_size = (self.sections * self.section_width, 500)
        self.window = None
        self.clock = None
        self.type_colors = None

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasiuspspom.farama.org/api/spaces/
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    def get_central_observation_space(self):
        """Returns the central observation space."""
        return self.central_observation_space

    @override
    def render(self):
        """Renders the environment as a top-down view of the beach.

        Each beach section is drawn as a column with an ocean strip on top and sand below.
        The agents standing in a section are drawn as circles, colored by their type, so that
        both objectives of the domain are visible at a glance: how crowded a section is (the
        occupation/capacity objective) and how well the agent types are mixed within it (the
        mixture objective). A section whose occupation exceeds its capacity is outlined in red.

        In "human" mode a window is opened and updated in place. In "rgb_array" mode the frame
        is returned as a `(height, width, 3)` uint8 numpy array, which is what
        `momaland/utils/generate_gif_image.py` uses to build the documentation GIFs.
        """
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

        if self.window is None:
            # Only initialize the subsystems actually used (display + font). pygame.init() also starts
            # the audio mixer and joystick subsystems, whose device enumeration adds ~0.4s of startup
            # (see Farama-Foundation/MOMAland#71). The display is initialized in the human branch only.
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("MO-Beach Domain")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
        if self.type_colors is None:
            self.type_colors = _generate_type_colors(len(self.type_distribution))

        title_font = pygame.font.SysFont("Arial", 16, bold=True)
        font = pygame.font.SysFont("Arial", 13)

        # Count the agents (and their types) standing in each section. We read directly from
        # self._state/self._types rather than from self.agents, because step() empties
        # self.agents on the terminal timestep before calling render().
        # Use len(self._state) rather than self.num_agents: the latter is len(self.agents), which is
        # 0 after step() empties self.agents on the terminal timestep (and beach defaults to a single
        # timestep, so the only render would otherwise be blank).
        agents_per_section = [[] for _ in range(self.sections)]
        for i in range(len(self._state)):
            agents_per_section[self._state[i]].append(self._types[i])

        sand_color = (237, 201, 175)
        ocean_color = (64, 164, 223)
        separator_color = (194, 178, 128)
        width, height = self.window_size
        ocean_height = int(height * 0.18)
        label_height = 56

        self.window.fill(sand_color)
        pygame.draw.rect(self.window, ocean_color, pygame.Rect(0, 0, width, ocean_height))

        # Type legend, drawn in the ocean strip.
        for t, color in enumerate(self.type_colors):
            lx = 12 + t * 90
            pygame.draw.circle(self.window, color, (lx + 8, ocean_height // 2), 7)
            self.window.blit(font.render(f"type {t}", True, (255, 255, 255)), (lx + 20, ocean_height // 2 - 8))

        agent_area_top = ocean_height + 8
        agent_area_bottom = height - label_height
        for s in range(self.sections):
            x0 = s * self.section_width
            capacity = self.resource_capacities[s]
            types_here = agents_per_section[s]
            consumption = len(types_here)
            over_capacity = consumption > capacity

            # Lay the agents out on a square-ish grid inside the section.
            if consumption > 0:
                cols = int(np.ceil(np.sqrt(consumption)))
                rows = int(np.ceil(consumption / cols))
                cell_w = (self.section_width - 20) / cols
                cell_h = (agent_area_bottom - agent_area_top) / rows
                radius = max(3, int(min(cell_w, cell_h) * 0.32))
                for k, t in enumerate(types_here):
                    c, r = k % cols, k // cols
                    cx = int(x0 + 10 + c * cell_w + cell_w / 2)
                    cy = int(agent_area_top + r * cell_h + cell_h / 2)
                    pygame.draw.circle(self.window, self.type_colors[t % len(self.type_colors)], (cx, cy), radius)
                    pygame.draw.circle(self.window, (40, 40, 40), (cx, cy), radius, 1)

            # Section border (red when over capacity) and labels.
            border_color = (200, 40, 40) if over_capacity else separator_color
            pygame.draw.rect(
                self.window,
                border_color,
                pygame.Rect(x0, ocean_height, self.section_width, height - ocean_height),
                4 if over_capacity else 1,
            )
            self.window.blit(title_font.render(f"Section {s}", True, (0, 0, 0)), (x0 + 8, height - label_height + 6))
            count_color = (200, 40, 40) if over_capacity else (0, 90, 0)
            self.window.blit(
                font.render(f"{consumption}/{capacity} agents", True, count_color),
                (x0 + 8, height - label_height + 28),
            )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment data which should not be kept around after the user is no longer using the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(), and step() can be called without issues.

        Returns the observations for each agent
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agents = self.possible_agents[:]
        self._types, self._state = self._init_state()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        section_consumptions, section_agent_types = self._get_stats()
        observations = {
            agent: self._get_obs(i, section_consumptions, section_agent_types) for i, agent in enumerate(self.agents)
        }
        self.episode_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _init_state(self):
        """Initializes the state of the environment. This is called by reset()."""
        types = random.choices(
            [i for i in range(len(self.type_distribution))],
            weights=self.type_distribution,
            k=self.num_agents,
        )

        if self.position_distribution is None:
            positions = [random.randint(0, self.sections - 1) for _ in self.agents]
        else:
            positions = random.choices(
                [i for i in range(self.sections)],
                weights=self.position_distribution,
                k=self.num_agents,
            )
        return types, positions

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

        # Apply actions and update system state
        for i, agent in enumerate(self.agents):
            act = actions[agent]
            self._state[i] = min(self.sections - 1, max(self._state[i] + MOVES[act], 0))

        section_consumptions, section_agent_types = self._get_stats()

        self.episode_num += 1

        env_termination = self.episode_num >= self.num_timesteps
        self.terminations = {agent: env_termination for agent in self.agents}
        reward_per_section = np.zeros((self.sections, NUM_OBJECTIVES), dtype=np.float32)

        if env_termination:
            if self.reward_mode == "individual":
                for i in range(self.sections):
                    lr_capacity = _local_capacity_reward(self.resource_capacities[i], section_consumptions[i])
                    lr_mixture = _local_mixture_reward(section_agent_types[i])
                    reward_per_section[i] = np.array([lr_capacity, lr_mixture])

            elif self.reward_mode == "team":
                g_capacity = _global_capacity_reward(self.resource_capacities, section_consumptions)
                g_mixture = _global_mixture_reward(section_agent_types)
                reward_per_section = np.array([[g_capacity, g_mixture]] * self.sections)

        # Obs: agent type, section id, section capacity, section consumption, % of agents of current type
        observations = {agent: None for agent in self.agents}
        # Note that agents only receive the reward after the last timestep
        rewards = {self.agents[i]: np.array([0, 0], dtype=np.float32) for i in range(self.num_agents)}

        for i, agent in enumerate(self.agents):
            observations[agent] = self._get_obs(i, section_consumptions, section_agent_types)
            rewards[agent] = reward_per_section[self._state[i]]

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_termination:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos

    @override
    def state(self) -> np.ndarray:
        """Returns the global observation of the beach.

        Returns: a 1D Numpy array with the following items in order:
        [agentX_section, agentX_type, ...,
        capacity, sectionY_consumption, sectionY_%_of_agents_of_current_type, ...]
        """
        # return np.array(self._types + self._state, dtype=np.int32)
        section_consumptions, section_agent_types = self._get_stats()
        global_obs = [self._get_obs(i, section_consumptions, section_agent_types) for i in range(len(self.agents))]
        global_obs = np.array(global_obs, dtype=np.float32).flatten()
        assert len(global_obs) == len(self.agents) * 5

        return np.array(global_obs, dtype=np.float32).flatten()

    def _get_obs(self, i, section_consumptions, section_agent_types):
        total_same_type = section_agent_types[self._state[i]][self._types[i]]
        t = total_same_type / section_consumptions[self._state[i]]
        obs = np.array(
            [
                self._types[i],
                self._state[i],
                self.resource_capacities[self._state[i]],
                section_consumptions[self._state[i]],
                t,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_stats(self):
        section_consumptions = np.zeros(self.sections)
        section_agent_types = np.zeros((self.sections, len(self.type_distribution)))

        for i in range(len(self.agents)):
            section_consumptions[self._state[i]] += 1
            section_agent_types[self._state[i]][self._types[i]] += 1
        return section_consumptions, section_agent_types


def _generate_type_colors(num_types):
    """Returns a list of `num_types` visually distinct RGB colors, evenly spaced around the hue wheel."""
    colors = []
    for i in range(num_types):
        hue = i / max(num_types, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def _global_capacity_reward(capacities, consumptions):
    global_capacity_r = 0
    for i in range(len(capacities)):
        global_capacity_r += _local_capacity_reward(capacities[i], consumptions[i])
    return global_capacity_r


def _local_capacity_reward(capacity, consumption):
    # TODO make capacity lookup table to save CPU!
    return consumption * np.exp(-consumption / capacity)


def _global_mixture_reward(section_agent_types):
    sum_local_mix = 0
    for i in range(len(section_agent_types)):
        sum_local_mix += _local_mixture_reward(section_agent_types[i])
    return sum_local_mix / len(section_agent_types)


def _local_mixture_reward(types):
    lr_mixture = 0
    if sum(types) > 0:
        lr_mixture = min(types) / sum(types)
    return lr_mixture
