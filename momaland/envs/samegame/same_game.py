"""MO-SameGame.

|--------------------|----------------------------------------------------|
| Actions            | Discrete                                           |
| Parallel API       | No                                                 |
| Manual Control     | No                                                 |
| Agents             | num_agents=1                                       |
| Action Shape       | (1,)                                               |
| Action Values      | Discrete(board_width=15 * board_height=15)         |
| Observation Shape  | (board_height=15, board_width=15, num_colors=5)    |
| Observation Values | [0,1]                                              |
| Reward Shape       | (num_objectives,)                                  |
"""

# Adapted from https://github.com/waldner/samepy/tree/master
# Copyright Davide Brini, 16/07/2014
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

from __future__ import annotations

import copy
import sys
from typing_extensions import override

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.logger import warn
from gymnasium.utils import EzPickle
from pettingzoo.utils import AgentSelector, wrappers

from momaland.utils.env import MOAECEnv


# RGB color for each tile value: index 0 is empty, 1-10 are the game colors.
_COLOR_RGB = [
    (236, 234, 228),
    (214, 64, 64),
    (66, 110, 205),
    (76, 175, 92),
    (232, 206, 60),
    (148, 82, 184),
    (235, 150, 45),
    (74, 195, 205),
    (214, 96, 170),
    (140, 200, 70),
    (150, 110, 80),
]


def env(**kwargs):
    """Returns the wrapped MOSameGame environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped AEC env
    """
    env = raw_env(**kwargs)
    # This wrapper terminates the game with the current player losing in case of illegal values.
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    # Asserts if the action given to step is outside the action space.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Checks if function calls or attribute access are in a disallowed order.
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """Returns the MOBreakthrough environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to create the `MOBreakthrough` environment.

    Returns:
        A raw env.
    """
    return MOSameGame(**kwargs)


class MOSameGame(MOAECEnv, EzPickle):
    """Multi-objective Multi-agent SameGame.

    MO-SameGame is a multi-objective, multi-agent variant of the single-player, single-objective turn-based puzzle
    game called SameGame.
    1 to 5 agents can play (default is 1), on a rectangular board with width and height from 3 to 30 squares (
    defaults are 15), which are initially filled with randomly colored tiles in 2 to 10 different colors (default is
    5). Players move in sequential order by selecting any tile in a group of at least 2 vertically and/or
    horizontally connected tiles of the same color. This group then disappears from the board. Tiles that were above
    the removed group "fall down" to close any vertical gaps; when entire columns of tiles become empty, all columns
    to the right move left to close the horizontal gap.
    Single-player, single-objective SameGame rewards the player with n^2 points for removing any group of n tiles.
    MO-SameGame can extend this in two ways. Agents can either only get points for their own actions, or all rewards
    can be shared. Additionally, points for every color can be counted as separate objectives, or they can be
    accumulated in a single objective like in the default game variant.

    ## Observation Space
    The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described
    below, and an `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section below.
    The main observation space is num_colors planes of a board_height * board_width grid (a board_height * board_width *
    num_colors tensor). Each plane represents the tiles of a specific color, and each location in the grid represents a
    location on the board. 1 indicates that a given location has a tile of the given plane's color, and 0
    indicates there is no tile of that color at that location (meaning that either the board location is empty, or filled
    by a tile of another color).

    ## Legal Actions Mask
    The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
    The `action_mask` is a binary vector where each index of the vector represents whether the represented action is legal
    or not; the action encoding is described in the Action Space section below.
    The `action_mask` will be all zeros for any agent except the one whose turn it is.

    ## Action Space
    The action space is the set of integers from 0 to board_width * board_height (exclusive). If the group connected to the
    tile at coordinates (x,y) is removed, this is encoded as the integer y * board_width + x.

    ## Rewards
    Rewards can be team rewards or individual rewards (default is individual).
    If color_rewards = False:
    Dimension 0: n^2 points for the removal of any group of size n.
    If color_rewards = True (default):
    Dimensions d=0 to d=num_objectives-1: n^2 points for the removal of any group of size n in color d+1.

    ## Starting State
    The starting board is filled with randomly colored tiles in 2 to 10 different colors (default is 5).

    ## Arguments
    - 'board_width': The width of the board (between 3 and 30)
    - 'board_height': The height of the board (between 3 and 30)
    - 'num_colors': The number of colors (between 2 and 10)
    - 'num_agents': The number of agents (between 1 and 5)
    - 'team_rewards': True = agents share all rewards, False = agents get individual rewards
    - 'color_rewards': True = agents get separate rewards for each color, False = agents get a single reward accumulating all colors
    - 'render_mode': The render mode

    ## Version History
    """

    metadata = {
        "render_modes": ["ansi", "human", "rgb_array"],
        "name": "mosame_game_v0",
        "is_parallelizable": False,
        "render_fps": 3,
    }

    BLANK = 0

    def __init__(
        self,
        board_width: int = 15,
        board_height: int = 15,
        num_colors: int = 5,
        num_agents: int = 1,
        team_rewards: bool = False,
        color_rewards: bool = True,
        render_mode: str | None = None,
    ):
        """Initializes the MOSameGame environment.

        Args:
            board_width: The width of the board (between 3 and 30)
            board_height: The height of the board (between 3 and 30)
            num_colors: The number of colors (between 2 and 10)
            num_agents: The number of agents (between 1 and 5)
            team_rewards: True = agents share all rewards, False = agents get individual rewards
            color_rewards: True = agents get separate rewards for each color, False = agents get a single reward accumulating all colors
            render_mode: The render mode
        """
        EzPickle.__init__(
            self,
            board_width,
            board_height,
            num_colors,
            num_agents,
            team_rewards,
            color_rewards,
            render_mode,
        )
        self.env = super().__init__()

        self.rng = np.random.default_rng()
        self.rng_initial_state = self.rng.__getstate__()
        self.gameinfo = {}
        self.gameinfo["ncolors"] = num_colors
        self.gameinfo["boardcols"] = board_width
        self.gameinfo["boardrows"] = board_height
        self.color_rewards = color_rewards
        if color_rewards:
            self.num_objectives = num_colors
        else:
            self.num_objectives = 1
        self.team_rewards = team_rewards
        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(r) for r in range(num_agents)]
        self.agents = self.possible_agents[:]
        self.move_count = 0
        self.max_move = board_height * board_width
        self._check_parameters()
        self._initialize_board()
        self.legal_moves = self._legal_moves()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        self.action_spaces = {agent: spaces.Discrete(self.max_move) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(board_height, board_width, self.gameinfo["ncolors"]),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(self.max_move,), dtype=np.int8),
                }
            )
            for agent in self.agents
        }
        self.reward_spaces = dict(
            zip(
                self.agents,
                [
                    spaces.Box(
                        low=0,
                        high=self._score(board_height * board_width),
                        shape=(self.num_objectives,),
                    )
                ]
                * len(self.agents),
            )
        )

        # pygame rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._cell = 30
        self._panel_w = 240
        self.window_size = (board_width * self._cell + self._panel_w, board_height * self._cell + 50)
        self.window = None
        self.clock = None
        # running score per agent across the episode (the env zeroes _cumulative_rewards each step)
        self._score_so_far = {agent: np.zeros(self.num_objectives) for agent in self.possible_agents}

    @override
    def render(self):
        """Renders the environment.

        In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            warn("You are calling render method without specifying any render mode.")
            return

        if self.render_mode == "ansi":
            self._print_board()
            return

        # human / rgb_array: draw the colored tile grid and a per-color score panel.
        cols = self.gameinfo["boardcols"]
        rows = self.gameinfo["boardrows"]
        cell = self._cell

        if self.window is None:
            # Only initialize the subsystems actually used (display + font). pygame.init() also starts
            # the audio mixer and joystick subsystems, whose device enumeration adds ~0.4s of startup
            # (see Farama-Foundation/MOMAland#71). The display is initialized in the human branch only.
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("MO-SameGame")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        font = pygame.font.SysFont("Arial", 14)
        small_font = pygame.font.SysFont("Arial", 12)

        self.window.fill((247, 245, 238))
        self.window.blit(title_font.render("MO-SameGame", True, (20, 20, 20)), (12, 12))

        board = self.gameinfo["board"][self.gameinfo["curmove"]]
        top = 40
        # board[col][row]: row 0 is drawn at the top, matching the ansi renderer.
        for col in range(cols):
            for row in range(rows):
                value = int(board[col][row])
                if value == MOSameGame.BLANK:
                    continue
                color = _COLOR_RGB[value] if value < len(_COLOR_RGB) else (0, 0, 0)
                rect = pygame.Rect(col * cell + 1, top + row * cell + 1, cell - 2, cell - 2)
                pygame.draw.rect(self.window, color, rect, border_radius=4)

        # Score panel.
        panel_x = cols * cell + 16
        self.window.blit(title_font.render("Scores", True, (20, 20, 20)), (panel_x, 14))
        y = 44
        for i, agent in enumerate(self.possible_agents):
            marker = ">" if agent == self.agent_selection else " "
            self.window.blit(font.render(f"{marker} agent {i}", True, (20, 20, 20)), (panel_x, y))
            y += 22
            score = self._score_so_far[agent]
            for o in range(self.num_objectives):
                # When color_rewards is on, each objective o is the tile color o+1.
                swatch = _COLOR_RGB[o + 1] if self.color_rewards and o + 1 < len(_COLOR_RGB) else (90, 90, 90)
                pygame.draw.rect(self.window, swatch, pygame.Rect(panel_x + 12, y, 14, 14))
                pygame.draw.rect(self.window, (80, 80, 80), pygame.Rect(panel_x + 12, y, 14, 14), 1)
                label = f"color {o + 1}" if self.color_rewards else "total"
                self.window.blit(small_font.render(f"{label}: {int(score[o])}", True, (20, 20, 20)), (panel_x + 34, y))
                y += 20
            self.window.blit(small_font.render(f"sum: {int(score.sum())}", True, (40, 40, 40)), (panel_x + 12, y))
            y += 26

        tiles_left = self.gameinfo["cellsleft"]["total"]
        self.window.blit(small_font.render(f"moves: {self.move_count}", True, (60, 60, 60)), (panel_x, y))
        self.window.blit(small_font.render(f"tiles left: {tiles_left}", True, (60, 60, 60)), (panel_x, y + 18))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    @override
    def observe(self, agent):
        board = self.gameinfo["board"][self.gameinfo["curmove"]]
        observation = np.stack(
            ([np.equal(board, color + 1) for color in range(self.gameinfo["ncolors"])]),
            axis=2,
        ).astype(np.int8)
        actions = self.legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros(self.max_move, "int8")
        for i in actions:
            action_mask[i] = 1
        return {"observation": observation, "action_mask": action_mask}

    @override
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @override
    def action_space(self, agent):
        return self.action_spaces[agent]

    @override
    def reward_space(self, agent):
        return self.reward_spaces[agent]

    def _legal_moves(
        self,
    ):
        """Returns a list of legal moves for the current player."""
        legal_moves = set()
        board = self.gameinfo["board"][self.gameinfo["curmove"]]
        for index, value in np.ndenumerate(board):
            x, y = index
            if board[x][y] != MOSameGame.BLANK and len(self._get_group(x, y)) > 1:
                legal_moves.add(self._move_to_int(x, y))
        return legal_moves

    def _move_to_int(self, x, y):
        """Converts x, y move coordinates to integer move encoding."""
        return y * self.gameinfo["boardrows"] + x

    def _int_to_move(self, move):
        """Converts integer move encoding to x, y move coordinates."""
        return move % self.gameinfo["boardrows"], move // self.gameinfo["boardrows"]

    def _check_parameters(self):
        """Checks constraints on game parameters."""
        if not (2 < self.gameinfo["boardrows"] < 31):
            raise ValueError("Board height cannot be set to %s - it has to be between 3 and 30." % self.gameinfo["boardrows"])
        if not (2 < self.gameinfo["boardcols"] < 31):
            raise ValueError("Board width cannot be set to %s - it has to be between 3 and 30." % self.gameinfo["boardcols"])
        if not (1 < self.gameinfo["ncolors"] < 11):
            raise ValueError(
                "Number of colors cannot be set to %s - it has to be between 2 and 10." % self.gameinfo["ncolors"]
            )
        if not (0 < self.num_agents < 6):
            raise ValueError("Number of agents cannot be set to %s - it has to be between 1 and 5." % self.num_agents)
        # print('Starting game with values: columns %s, rows %s, colors %s, agents %s' % (
        #     self.gameinfo['boardcols'], self.gameinfo['boardrows'], self.gameinfo['ncolors'], self.num_agents))

    def _print_err(self, msg):
        sys.stderr.write(msg + "\n")

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment
        data which should not be kept around after the user is no longer using the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    @override
    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        # assert valid move
        assert action in self.legal_moves, "played illegal move."

        # make the move
        x, y = self._int_to_move(action)
        curgroup = self._get_group(x, y)
        self._remove_group(curgroup)
        self.move_count += 1

        # handle the rewards
        action_score = self.gameinfo["score"][self.gameinfo["curmove"]]
        action_color = self.gameinfo["color"][self.gameinfo["curmove"]]
        self.rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        if self.color_rewards:
            index = action_color - 1
        else:
            index = 0
        agent = self.agent_selection
        if self.team_rewards:
            for a in self.agents:
                self.rewards[a][index] = action_score
        else:
            self.rewards[agent][index] = action_score
        if self._game_over():
            self.terminations = {agent: True for agent in self.agents}
        self._cumulative_rewards[agent] = np.zeros(self.num_objectives)
        self._accumulate_rewards()

        # track a running per-agent score for rendering (_cumulative_rewards is zeroed each step)
        for a in self.agents:
            self._score_so_far[a] = self._score_so_far[a] + self.rewards[a]

        # select the next agent
        self.agent_selection = self._agent_selector.next()
        self.legal_moves = self._legal_moves()
        if self.render_mode == "human":
            self.render()

    @override
    def reset(self, seed=None, options=None):
        if seed is None:
            self.rng.__setstate__(self.rng_initial_state)
        else:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.move_count = 0
        self._score_so_far = {agent: np.zeros(self.num_objectives) for agent in self.possible_agents}
        self._initialize_board()
        self.legal_moves = self._legal_moves()

    def _initialize_board(self):
        """Initializes the board."""
        self.gameinfo["board"] = {}
        self.gameinfo["score"] = {}
        self.gameinfo["color"] = {}

        # to support undo/redo  # TODO remove undo/redo
        self.gameinfo["curmove"] = 0
        self.gameinfo["maxmove"] = 0

        # actually fill the board
        board = []
        for c in range(self.gameinfo["boardcols"]):
            col = []
            for r in range(self.gameinfo["boardrows"]):
                color = self.rng.integers(1, self.gameinfo["ncolors"] + 1)
                col.append(color)
            board.append(col)
        self.gameinfo["board"][self.gameinfo["curmove"]] = board

        self._calculate_all_groups()
        self.gameinfo["score"][self.gameinfo["curmove"]] = 0
        self.gameinfo["color"][self.gameinfo["curmove"]] = 0
        self.gameinfo["lastnremoved"] = 0
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def _game_won(self):
        """Checks if the game was won, i.e. if all tiles have been removed from the board."""
        return self.gameinfo["cellsleft"]["total"] == 0

    def _game_over(self):
        """Checks if the game is over, i.e. if there are no groups of size >=2 anymore that can be removed."""
        return self.gameinfo["maxgroupsize"] < 2

    def _print_board(self):
        """Prints the board in a human-readable format."""
        for row in range(self.gameinfo["boardrows"]):
            for col in range(self.gameinfo["boardcols"]):
                if self.gameinfo["board"][self.gameinfo["curmove"]][col][row] == MOSameGame.BLANK:
                    print(".  ", end="")
                else:
                    print(
                        self.gameinfo["board"][self.gameinfo["curmove"]][col][row],
                        " ",
                        end="",
                    )
            print()
        print()

    def _score(self, removed_cells):
        """Returns the score for an action that removed removed_cells cells."""
        return removed_cells**2

    def _get_immediate_neighbors(self, x, y):
        """Returns the immediate neighbors of the tile at (x,y) that have the same color."""
        board = self.gameinfo["board"][self.gameinfo["curmove"]]
        color = board[x][y]
        neighbors = []

        # left neighbor
        if x > 0 and board[x - 1][y] == color:
            neighbors.append((x - 1, y))
        # right neighbor
        if x < self.gameinfo["boardcols"] - 1 and board[x + 1][y] == color:
            neighbors.append((x + 1, y))
        # upper neighbor
        if y > 0 and board[x][y - 1] == color:
            neighbors.append((x, y - 1))
        # lower neighbor
        if y < self.gameinfo["boardrows"] - 1 and board[x][y + 1] == color:
            neighbors.append((x, y + 1))

        return neighbors

    def _calculate_group(self, cellx, celly):
        """Returns the entire group of the tile at (x,y) that has the same color."""
        to_process = {(cellx, celly)}
        processed = set()

        while len(to_process) > 0:
            # extract one element
            x, y = tuple(to_process)[0]
            for n in self._get_immediate_neighbors(x, y):
                if n not in processed:
                    to_process.add(n)
            to_process.remove((x, y))
            processed.add((x, y))

        return tuple(sorted(list(processed)))

    def _calculate_all_groups(self):
        """Calculates all groups with the same color on the board."""
        maxgroupsize = 0
        board = self.gameinfo["board"][self.gameinfo["curmove"]]
        self.gameinfo["cellsleft"] = {}
        self.gameinfo["groupsleft"] = {}
        for colorNo in range(self.gameinfo["ncolors"]):
            self.gameinfo["cellsleft"][colorNo] = 0
            self.gameinfo["groupsleft"][colorNo] = {}
        self.gameinfo["cellsleft"]["total"] = 0
        self.gameinfo["groupsleft"]["total"] = {}

        processedcells = {}
        for col in range(self.gameinfo["boardcols"]):
            for row in range(self.gameinfo["boardrows"]):
                colorNo = board[col][row]
                if colorNo == MOSameGame.BLANK:
                    continue
                else:
                    colorNo -= 1
                    if not (col, row) in processedcells:
                        group = self._calculate_group(col, row)
                        groupsize = len(group)
                        # to avoid processing other cells belonging
                        # to the same group
                        for cell in group:
                            processedcells[cell] = None
                        if group not in self.gameinfo["groupsleft"][colorNo]:
                            if groupsize > 1:
                                self.gameinfo["groupsleft"][colorNo][group] = None
                            self.gameinfo["cellsleft"][colorNo] += groupsize
                            if groupsize > 1:
                                self.gameinfo["groupsleft"]["total"][group] = None
                            self.gameinfo["cellsleft"]["total"] += groupsize
                            if groupsize > maxgroupsize:
                                maxgroupsize = groupsize

        self.gameinfo["maxgroupsize"] = maxgroupsize

    def _get_group(self, x, y):
        """Assuming all groups have been calculated, returns the group tile (x,y) belongs to."""
        for group in self.gameinfo["groupsleft"]["total"]:
            if (x, y) in group:
                return group
        return ((x, y),)

    def _remove_group(self, group):
        """Removes a group of tiles of same color and computes the new board status. A copy of the current status is made, and the copy is updated and saved."""
        # if the group is a single cell, do nothing
        if len(group) == 1:
            return
        nremoved = len(group)

        # consider the new move
        score = self.gameinfo["score"][self.gameinfo["curmove"]]
        newstate = copy.deepcopy(self.gameinfo["board"][self.gameinfo["curmove"]])
        self.gameinfo["curmove"] += 1
        self.gameinfo["board"][self.gameinfo["curmove"]] = newstate
        self.gameinfo["score"][self.gameinfo["curmove"]] = score
        newstate = self.gameinfo["board"][self.gameinfo["curmove"]]

        # to handle redo
        self.gameinfo["maxmove"] = self.gameinfo["curmove"]

        # first, find which columns are affected by the removal.
        # at the same time, blank out the removed cells.
        affectedcols = {}
        cellx, celly = group[0]
        group_color = newstate[cellx][celly]
        self.gameinfo["color"][self.gameinfo["curmove"]] = group_color
        for n in group:
            col, row = n
            newstate[col][row] = MOSameGame.BLANK
            affectedcols[col] = None

        # second, fill the gaps in the affected columns
        # by moving down things

        for col in affectedcols:
            # create a column of empty cells
            occupiedrows = [MOSameGame.BLANK] * self.gameinfo["boardrows"]
            for row in range(0, self.gameinfo["boardrows"]):
                if newstate[col][row] != MOSameGame.BLANK:
                    occupiedrows.append(newstate[col][row])
            # get last self.gameinfo['boardrows'] elements, which should
            # correspond to the actually used cells
            newstate[col] = occupiedrows[-self.gameinfo["boardrows"] :]

        # third step: shift columns left if any column is empty
        # same logic as used for filling row gaps, but reversed

        occupiedcols = []
        for col in range(0, self.gameinfo["boardcols"]):
            # if lowest cell is not blank, column is not empty
            if newstate[col][self.gameinfo["boardrows"] - 1] != MOSameGame.BLANK:
                occupiedcols.append(newstate[col])

        occupiedcols.extend([[MOSameGame.BLANK] * self.gameinfo["boardrows"] for x in range(self.gameinfo["boardcols"])])
        newstate = occupiedcols[: self.gameinfo["boardcols"]]

        # this becomes the current state
        self.gameinfo["board"][self.gameinfo["curmove"]] = newstate
        # finally, recalculate group info
        self._calculate_all_groups()
        self.gameinfo["lastnremoved"] = nremoved
        self.gameinfo["score"][self.gameinfo["curmove"]] = self._score(self.gameinfo["lastnremoved"])
