"""Ingenious environment.

|--------------------|--------------------------------------------------------------|
| Actions            | Discrete                                                     |
| Parallel API       | No                                                           |
| Manual Control     | No                                                           |
| Agents             | num_agents=2                                                 |
| Action Shape       | (1,)                                                         |
| Action Values      | Discrete(size depends on board size and rack size: there     |
|                    |  is one integer encoding the placement of each rack tile     |
|                    |  on each board hex in each possible direction.)              |
| Observations       | Observations are dicts with three entries:                   |
|                    |  "board": array with size (2*board_size-1, 2*board_size-1)   |
|                    |  containing values from 0 to num_colors;                     |
|                    |  "racks": for each observable agent, an array of length      |
|                    |  rack_size containing pairs of values from 0 to num_colors;  |
|                    |  "scores": for all agents, their scores in all num_colors    |
|                    |  objectives as values from 0 to max_score.                   |
| Reward Shape       | (num_colors=6,)                                              |

This environment is based on the Ingenious game: https://boardgamegeek.com/boardgame/9674/ingenious

The game's original rules support multiple players collecting scores in multiple colors, which we define as the
objectives of the game: for example (red=5, green=2, blue=9). The goal in the original game is to maximize the
minimum score over all colors (2 in the example above), however we leave the utility wrapper up to the users and only
return the vectorial score on each color dimension (5,2,9).


### Observation Space

The observation is a dictionary which contains an 'observation' element which is the usual RL observation,
and an 'action_mask' which holds the legal moves, described in the Legal Actions Mask section below.

The 'observation' element itself is a dictionary with three entries: 'board' is representing the hexagonal board as
an array of size (2*board_size-1, 2*board_size-1) with integer entries from 0 (empty hex) to num_colors (tiles of
different colors). 'racks' represents for each observable agent - by default only the acting agent, if fully_obs=True
all agents - their tiles rack as an array of size rack_size containing pairs of integers (each pair is a tile) from 0
to num_colors. 'scores' represents for all agents their current scores in all num_colors objectives, as integers from
0 to max_score.


#### Legal Actions Mask

The legal moves available to the current agent are found in the 'action_mask' element of the dictionary observation.
The 'action_mask' is a binary vector where each index of the vector represents whether the represented action is legal
or not; the action encoding is described in the Action Space section below.
The 'action_mask' shows only the current agent's legal moves.


### Action Space

The action space depends on board size and rack size: It contains one integer for each possible placement of any of
the player's rack tiles (rack_size parameter) on any board hex (board_size parameter) in every possible direction.


### Rewards

The agents can collect a separate score in each available color. These scores are the num_colors different reward
dimensions.


### Version History

"""

import functools
import math
import random
from typing_extensions import override

import numpy as np
import pygame
from gymnasium.logger import warn
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers

from momaland.envs.ingenious.ingenious_base import (
    ALL_COLORS,
    Hex2ArrayLocation,
    IngeniousBase,
)
from momaland.utils.env import MOAECEnv


# RGB color for each tile value on the board: 0 = empty, 1-6 = the game colors (in ALL_COLORS order:
# red, green, blue, orange, yellow, purple).
_COLOR_RGB = [
    (228, 226, 220),
    (214, 64, 64),
    (76, 175, 92),
    (66, 110, 205),
    (235, 150, 45),
    (232, 206, 60),
    (148, 82, 184),
]


def _hex_corners(cx, cy, size):
    """Returns the 6 corner points of a pointy-top hexagon centered at (cx, cy)."""
    return [
        (cx + size * math.cos(math.radians(60 * i - 30)), cy + size * math.sin(math.radians(60 * i - 30))) for i in range(6)
    ]


def env(**kwargs):
    """Returns the wrapped Ingenious environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped AEC env
    """
    env = raw_env(**kwargs)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the Ingenious environment."""
    return Ingenious(**kwargs)


class Ingenious(MOAECEnv, EzPickle):
    """Environment for the Ingenious board game."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "moingenious_v0",
        "is_parallelizable": False,
        "render_fps": 3,
    }

    def __init__(
        self,
        num_agents: int = 2,
        rack_size: int = 6,
        num_colors: int = 6,
        board_size: int = None,
        reward_mode: str = "competitive",
        fully_obs: bool = False,
        render_mode: bool = None,
    ):
        """Initializes the Ingenious environment.

        Args:
            num_agents (int): The number of agents (between 2 and 6). Default is 2.
            rack_size (int): The number of tiles each player keeps in their rack (between 2 and 6). Default is 6.
            num_colors (int): The number of colors (objectives) in the game (between 2 and 6). Default is 6.
            board_size (int): The size of one side of the hexagonal board (between 3 and 10). By default the size is set
             to n+4 where n is the number of agents.
            reward_mode (str): Can be set to "competitive" (individual rewards for all agents), "collaborative" (shared
            rewards for all agents), or "two_teams" (rewards shared within two opposing teams; num_agents needs to be
            even). Default is "competitive".
            fully_obs (bool): Fully observable game mode, i.e. the racks of all players are visible. Default is False.
            render_mode (str): The rendering mode. Default: None
        """
        EzPickle.__init__(
            self,
            num_agents,
            rack_size,
            num_colors,
            board_size,
            reward_mode,
            fully_obs,
            render_mode,
        )
        self.num_colors = num_colors
        self.init_draw = rack_size
        self.max_score = 18  # max score in score board for one certain color.
        assert reward_mode in {
            "competitive",
            "collaborative",
            "two_teams",
        }, "reward_mode has to be one element in {'competitive','collaborative','two_teams'}"
        self.reward_mode = reward_mode
        self.fully_obs = fully_obs

        if self.reward_mode == "two_teams":
            assert num_agents % 2 == 0, "Number of players must be even if reward_mode is two_teams."
            self.max_score = self.max_score * (num_agents / 2)
        elif self.reward_mode == "collaborative":
            self.max_score = self.max_score * num_agents

        if board_size is None:
            self.board_size = {2: 6, 3: 7, 4: 8, 5: 9, 6: 10}.get(num_agents)
        else:
            self.board_size = board_size

        self.game = IngeniousBase(
            num_agents=num_agents,
            rack_size=self.init_draw,
            num_colors=self.num_colors,
            board_size=self.board_size,
            max_score=self.max_score,
        )

        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self._cumulative_rewards = {agent: np.zeros(self.num_colors) for agent in self.agents}
        self.refresh_cumulative_reward = True
        self.render_mode = render_mode

        self.observation_spaces = {
            i: Dict(
                {
                    "observation": Dict(
                        {
                            "board": Box(
                                0, len(ALL_COLORS), shape=(2 * self.board_size - 1, 2 * self.board_size - 1), dtype=np.float32
                            ),
                            "racks": (
                                Box(0, self.num_colors, shape=(num_agents, self.init_draw, 2), dtype=np.int32)
                                if self.fully_obs
                                else Box(0, self.num_colors, shape=(self.init_draw, 2), dtype=np.int32)
                            ),
                            "scores": Box(0, self.game.max_score, shape=(num_agents, self.num_colors), dtype=np.int32),
                        }
                    ),
                    "action_mask": Box(low=0, high=1, shape=(len(self.game.masked_action),), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.action_spaces = dict(zip(self.agents, [Discrete(len(self.game.masked_action))] * num_agents))

        # The reward for each move is the difference between the previous and current score.
        self.reward_spaces = dict(zip(self.agents, [Box(0, self.game.max_score, shape=(self.num_colors,))] * num_agents))

        # pygame rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = (820, 560)
        self.window = None
        self.clock = None

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
        return self.reward_spaces[agent]

    @override
    def render(self):
        """Renders the hexagonal Ingenious board and the per-color scores.

        The board is drawn as a hexagon of hexagonal tiles, each filled with its placed color (empty
        cells are grey). The right panel is the scoreboard: for every agent it shows the score reached
        in each color - these colors are the objectives of the game - together with the minimum over
        all colors, which is the quantity the original game asks players to maximize. The acting
        agent's rack of tiles is drawn underneath the board.

        In "human" mode a window is opened and updated in place. In "rgb_array" mode the frame is
        returned as a `(height, width, 3)` uint8 numpy array for GIF generation.
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
                pygame.display.set_caption("MO-Ingenious")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        font = pygame.font.SysFont("Arial", 14)
        small_font = pygame.font.SysFont("Arial", 12)

        board_size = self.game.board_size

        # Lay the board out by computing unit-size pixel coordinates for every hex, then scaling
        # them to fit the board area (between the title and the rack) and centering on the bounding
        # box - this keeps the whole hexagon on screen for any board size.
        board_w = 560
        margin_top, rack_h = 46, 84
        avail_w, avail_h = board_w - 20, self.window_size[1] - margin_top - rack_h
        unit = {hx: (math.sqrt(3) * (hx.q + hx.r / 2), 1.5 * hx.r) for hx in self.game.board_hex}
        xs = [p[0] for p in unit.values()]
        ys = [p[1] for p in unit.values()]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        # +2 padding (one hex diameter in unit coordinates) so border hexes are not clipped.
        hex_size = min(avail_w / (max_x - min_x + 2), avail_h / (max_y - min_y + 2))
        origin_x = board_w / 2 - hex_size * (min_x + max_x) / 2
        origin_y = margin_top + avail_h / 2 - hex_size * (min_y + max_y) / 2

        self.window.fill((247, 245, 238))
        self.window.blit(title_font.render("MO-Ingenious", True, (20, 20, 20)), (16, 12))

        # Board hexes.
        board_area = (board_w, self.window_size[1])
        for hx in self.game.board_hex:
            cx, cy = origin_x + hex_size * unit[hx][0], origin_y + hex_size * unit[hx][1]
            ax, ay = Hex2ArrayLocation(hx, board_size)
            color_val = int(self.game.board_array[ax][ay])
            color = _COLOR_RGB[color_val] if 0 <= color_val < len(_COLOR_RGB) else (0, 0, 0)
            corners = _hex_corners(cx, cy, hex_size * 0.95)
            pygame.draw.polygon(self.window, color, corners)
            pygame.draw.polygon(self.window, (90, 90, 90), corners, 1)

        # Scoreboard panel.
        panel_x = board_area[0] + 10
        self.window.blit(title_font.render("Scores", True, (20, 20, 20)), (panel_x, 14))
        row_y = 44
        for i, agent in enumerate(self.possible_agents):
            marker = ">" if agent == self.agent_selection else " "
            self.window.blit(font.render(f"{marker} agent {i}", True, (20, 20, 20)), (panel_x, row_y))
            row_y += 22
            scores = self.game.score[agent]
            for color_val in ALL_COLORS[: self.num_colors]:
                score = scores[color_val]
                pygame.draw.rect(self.window, _COLOR_RGB[color_val], pygame.Rect(panel_x + 12, row_y, 14, 14))
                pygame.draw.rect(self.window, (80, 80, 80), pygame.Rect(panel_x + 12, row_y, 14, 14), 1)
                # A small bar proportional to the score.
                bar_w = int((score / self.game.max_score) * 120)
                pygame.draw.rect(self.window, _COLOR_RGB[color_val], pygame.Rect(panel_x + 34, row_y, bar_w, 14))
                self.window.blit(small_font.render(str(int(score)), True, (20, 20, 20)), (panel_x + 160, row_y))
                row_y += 18
            min_score = min(scores[c] for c in ALL_COLORS[: self.num_colors])
            self.window.blit(small_font.render(f"min (utility): {int(min_score)}", True, (60, 60, 60)), (panel_x + 12, row_y))
            row_y += 26

        # Acting agent's rack, drawn as pairs of colored circles under the board.
        rack = self.game.p_tiles.get(self.agent_selection, [])
        rack_y = board_area[1] - 60
        self.window.blit(
            small_font.render(f"rack of agent {self.possible_agents.index(self.agent_selection)}:", True, (60, 60, 60)),
            (16, rack_y - 18),
        )
        for t, tile in enumerate(rack):
            tx = 20 + t * 70
            for k, c in enumerate(tile):
                c = int(c)
                color = _COLOR_RGB[c] if 0 <= c < len(_COLOR_RGB) else (0, 0, 0)
                pygame.draw.circle(self.window, color, (tx + k * 22, rack_y), 11)
                pygame.draw.circle(self.window, (80, 80, 80), (tx + k * 22, rack_y), 11, 1)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    @override
    def close(self):
        """Closes the rendering window."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    @override
    def reset(self, seed=None, options=None):
        """Reset needs to initialize the `agents` attribute and must set up the environment so that render(),
        and step() can be called without issues."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.game.reset_game(seed)
        self.agents = self.possible_agents[:]
        obs = {agent: self.observe(agent) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.game.agent_selector]
        self.agent_selection = self.agents[self.game.agent_selector]
        self.rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        self.refresh_cumulative_reward = True
        return obs, self.infos

    @override
    def step(self, action):
        """Steps in the environment.

        Args:
            action: action of the active agent
        """
        current_agent = self.agent_selection

        if self.terminations[current_agent] or self.truncations[current_agent]:
            return self._was_dead_step(action)
        self.rewards = {agent: np.zeros(self.num_colors, dtype="float64") for agent in self.agents}
        if self.refresh_cumulative_reward:
            self._cumulative_rewards[current_agent] = np.zeros(self.num_colors, dtype="float64")

        # update current agent
        if not self.game.end_flag:
            prev_rewards = np.array(list(self.game.score[current_agent].values()))
            self.game.set_action_index(action)
            current_rewards = np.array(list(self.game.score[current_agent].values()))
            self.rewards[current_agent] = current_rewards - prev_rewards

        if self.game.end_flag:
            self.terminations = {agent: True for agent in self.agents}

        # update teammate score (copy current agent's score to teammates)
        if self.reward_mode != "competitive":
            index_current_agent = self.agents.index(current_agent)
            for i in range(0, self.num_agents):
                if self.reward_mode == "two_teams":
                    # in two_team mode, players who are teammates of the current agent get the same reward and score
                    if i != index_current_agent and i % 2 == index_current_agent % 2:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]
                elif self.reward_mode == "collaborative":
                    # in collaborative mode, every player gets the same reward and score
                    if i != index_current_agent:
                        agent = self.agents[i]
                        self.game.score[agent] = self.game.score[current_agent]
                        self.rewards[agent] = self.rewards[current_agent]

        self._accumulate_rewards()

        # update to next agent
        self.agent_selection = self.agents[self.game.agent_selector]

        if self.agent_selection != current_agent:
            self.refresh_cumulative_reward = True
        else:
            self.refresh_cumulative_reward = False

        if self.render_mode == "human":
            self.render()

    @override
    def observe(self, agent):
        board_vals = np.array(self.game.board_array, dtype=np.float32)
        if self.fully_obs:
            p_tiles = np.array([item for item in self.game.p_tiles.values()], dtype=np.int32)
        else:
            p_tiles = np.array(self.game.p_tiles[agent], dtype=np.int32)
        tmp = []
        for agent_score in self.game.score.values():
            tmp.append([score for score in agent_score.values()])
        p_score = np.array(tmp, dtype=np.int32)
        observation = {"board": board_vals, "racks": p_tiles, "scores": p_score}
        action_mask = np.array(self.game.return_action_list(), dtype=np.int8)
        return {"observation": observation, "action_mask": action_mask}
