"""MO-Connect Four.

|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | No                                               |
| Manual Control     | No                                               |
| Agents             | 2                                                |
| Action Shape       | (1,)                                             |
| Action Values      | Discrete(board_width=7)                          |
| Observation Shape  | (board_height=6, board_width=7, 2)               |
| Observation Values | [0,1]                                            |
| Reward Shape       | (2,) or (2+board_width,)                         |
"""

from __future__ import annotations

import os
from typing_extensions import override

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo.utils import AgentSelector, wrappers

from momaland.utils.env import MOAECEnv


def get_image(path):
    """Load an image into a pygame surface."""
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image = pygame.image.load(cwd + "/" + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    """Returns the wrapped MOConnect4 environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to the raw_env function

    Returns:
        A fully wrapped AEC env
    """
    env = raw_env(**kwargs)
    # This wrapper terminates the game with the current player losing in case of illegal values.
    # env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    # Asserts if the action given to step is outside of the action space.
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Checks if function calls or attribute access are in a disallowed order.
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """Returns the MOConnect4 environment in `AEC` format.

    Args:
        **kwargs: keyword args to forward to create the `MOConnect4` environment.

    Returns:
        A raw env.
    """
    return MOConnect4(**kwargs)


class MOConnect4(MOAECEnv, EzPickle):
    """Multi-objective Connect Four.

    MO-Connect4 is a multi-objective variant of the two-player, single-objective turn-based board game Connect 4.
    In Connect 4, players can win by connecting four of their tokens vertically, horizontally or diagonally. The players
    drop their respective token in a column of a standing board (of width 7 and height 6 by default), where each token will
    fall until it reaches the bottom of the column or lands on top of an existing token.
    Players cannot place a token in a full column, and the game ends when either a player has made a sequence of 4 tokens,
    or when all columns have been filled (draw).
    MO-Connect4 extends this game with a second objective that incentivizes faster wins, and optionally the additional
    (conflicting) objectives of having more tokens than the opponent in every column. Additionally, width and height of the
    board can be set to values from 4 to 20.

    ## Observation Space
    The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described
    below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section below.
    The main observation space is 2 planes of a board_height * board_width grid (a board_height * board_width * 2 tensor).
    Each plane represents a specific agent's tokens, and each location in the grid represents the placement of the
    corresponding agent's token. 1 indicates that the agent has a token placed in the given location, and 0 indicates they
    do not have a token in that location (meaning that either the cell is empty, or the other agent has a token in that
    location).

    ## Legal Actions Mask
    The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
    The `action_mask` is a binary vector where each index of the vector represents whether the represented action is legal
    or not; the action encoding is described in the Action Space section below.
    The `action_mask` will be all zeros for any agent except the one whose turn it is.

    ## Action Space
    The action space is the set of integers from 0 to board_width (exclusive), where the number represents which column
    a token should be dropped in.

    ## Rewards
    Dimension 0: If an agent successfully connects four of their tokens, they will be rewarded 1 point. At the same time,
    the opponent agent will be awarded -1 point. If the game ends in a draw, both players are rewarded 0.
    Dimension 1: If an agent wins, they get a reward of 1-(move_count/board_size) to incentivize faster wins. The losing opponent gets the negated reward. In case of a draw, both agents get 0.
    Dimension 2 to board_width+1 (default 8): (optional) If at game end, an agent has more tokens than their opponent in
    column X, they will be rewarded 1 point in reward dimension 2+X. The opponent agent will be rewarded -1 point. If the
    column has an equal number of tokens from both players, both players are rewarded 0.

    ## Starting State
    The game starts with an empty board.

    ## Arguments
    - 'render_mode': The mode to render with. Can be 'human' or 'rgb_array'.
    - 'screen_scaling': The factor by which to scale the screen.
    - 'board_width': The width of the board (from 4 to 20)
    - 'board_height': The height of the board (from 4 to 20)
    - 'column_objectives': Whether to use column objectives or not (without them, there are 2 objectives. With them, there are 2+board_width objectives)

    ## Version History
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "moconnect4_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        screen_scaling: int = 9,
        board_width: int = 7,
        board_height: int = 6,
        column_objectives: bool = True,
    ):
        """Initializes a new MOConnect4 environment.

        Args:
            render_mode: The mode to render with. Can be 'human' or 'rgb_array'.
            screen_scaling: The factor by which to scale the screen.
            board_width: The width of the board (from 4 to 20)
            board_height: The height of the board (from 4 to 20)
            column_objectives: Whether to use column objectives or not (without them, there are 2 objectives. With them, there are 2+board_width objectives)
        """
        EzPickle.__init__(
            self,
            render_mode,
            screen_scaling,
            board_width,
            board_height,
            column_objectives,
        )
        self.env = super().__init__()

        if not (4 <= board_width <= 20):
            raise ValueError("Config parameter board_width must be between 4 and 20.")

        elif not (4 <= board_height <= 20):
            raise ValueError("Config parameter board_height must be between 4 and 20.")

        self.column_objectives = column_objectives
        self.screen = None
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = board_height * board_width
        self.board = [0] * self.board_size
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.num_objectives = 2 + board_width if column_objectives else 2
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.move_count = 0

        self.action_spaces = {agent: spaces.Discrete(board_width) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(board_height, board_width, len(self.agents)),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(board_width,), dtype=np.int8),
                }
            )
            for agent in self.agents
        }
        self.reward_spaces = dict(
            zip(
                self.agents,
                [spaces.Box(low=-1, high=1, shape=(self.num_objectives,))] * len(self.agents),
            ),
            dtype=np.float32,
        )

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

    @override
    def observe(self, agent):
        board_vals = np.array(self.board).reshape(self.board_height, self.board_width)
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2
        cur_p_board = np.equal(board_vals, cur_player + 1)
        opp_p_board = np.equal(board_vals, opp_player + 1)
        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        legal_moves = self._legal_moves() if agent == self.agent_selection else []
        action_mask = np.zeros(self.board_width, "int8")
        for i in legal_moves:
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

    def _legal_moves(self):
        """Returns a list of legal moves for the current player."""
        return [i for i in range(self.board_width) if self.board[i] == 0]

    @override
    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        # assert valid move
        assert self.board[0 : self.board_width][action] == 0, "played illegal move."

        # make the move
        agent = self.agent_selection
        piece = self.agents.index(agent) + 1
        for i in filter(lambda x: x % self.board_width == action, range(self.board_size - 1, -1, -1)):
            if self.board[i] == 0:
                self.board[i] = piece
                self.move_count += 1
                break

        # handle the rewards
        next_agent = self._agent_selector.next()
        winner = self.check_for_winner()
        self.rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        if winner:
            self.rewards[agent][0] = 1
            self.rewards[next_agent][0] = -1
            self.rewards[agent][1] = 1 - (self.move_count / self.board_size)
            self.rewards[next_agent][1] = -(1 - (self.move_count / self.board_size))
        # check if there is a tie
        if winner or all(x in [1, 2] for x in self.board):
            if self.column_objectives:
                self._assign_column_rewards(agent, next_agent)
            self.terminations = {agent: True for agent in self.agents}
        self._cumulative_rewards[agent] = np.zeros(self.num_objectives, dtype=np.float32)
        self._accumulate_rewards()

        # select the next agent
        self.agent_selection = next_agent
        if self.render_mode == "human":
            self.render()

    def _assign_column_rewards(self, agent, opp_agent):
        """Assigns rewards for columns based on who has more tokens in each column."""
        for i in range(self.board_width):
            agent_with_more_tokens = self.more_tokens_in_column(i)
            if agent_with_more_tokens == agent:
                self.rewards[agent][2 + i] += 1
                self.rewards[opp_agent][2 + i] -= 1
            elif agent_with_more_tokens == opp_agent:
                self.rewards[agent][2 + i] -= 1
                self.rewards[opp_agent][2 + i] += 1

    @override
    def reset(self, seed=None, options=None):
        self.board = [0] * (self.board_height * self.board_width)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: np.zeros(self.num_objectives, dtype=np.float32) for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_objectives, dtype=np.float32) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.move_count = 0

    @override
    def render(self):  # TODO adapt to different board sizes, low priority
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        screen_width = 99 * self.screen_scaling
        screen_height = 86 / 99 * screen_width

        if self.screen is None:
            if self.render_mode == "human":
                # Only initialize the display subsystem on the human render path. pygame.init() also
                # starts the unused audio mixer and joystick subsystems, whose device enumeration adds
                # ~0.4s of startup (see Farama-Foundation/MOMAland#71).
                pygame.display.init()
                pygame.display.set_caption("Connect Four")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        # Load and scale all of the necessary images
        size_cap = min(screen_width, screen_height)
        tile_size = (size_cap * 91 / 99) / max(self.board_width, self.board_height)

        red_chip = get_image(os.path.join("img", "C4RedPiece.png"))
        red_chip = pygame.transform.scale(red_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13))))

        black_chip = get_image(os.path.join("img", "C4BlackPiece.png"))
        black_chip = pygame.transform.scale(black_chip, (int(tile_size * (9 / 13)), int(tile_size * (9 / 13))))

        cell_img = get_image(os.path.join("img", "c4-empty.png"))
        cell_img = pygame.transform.scale(cell_img, (int(tile_size), int(tile_size)))

        for i in range(0, self.board_height * self.board_width):
            self.screen.blit(
                cell_img,
                (
                    (i % self.board_width) * (tile_size) + (tile_size * (4 / 13)),
                    int(i / self.board_width) * (tile_size) + (tile_size * (4 / 13)),
                ),
            )

        # Blit the necessary chips and their positions
        for i in range(0, self.board_height * self.board_width):
            if self.board[i] == 1:
                self.screen.blit(
                    red_chip,
                    (
                        (i % self.board_width) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / self.board_width) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )
            elif self.board[i] == 2:
                self.screen.blit(
                    black_chip,
                    (
                        (i % self.board_width) * (tile_size) + (tile_size * (6 / 13)),
                        int(i / self.board_width) * (tile_size) + (tile_size * (6 / 13)),
                    ),
                )

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return np.transpose(observation, axes=(1, 0, 2)) if self.render_mode == "rgb_array" else None

    @override
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def more_tokens_in_column(self, column):
        """Returns the agent with more tokens in the given column, or None if they are equal."""
        piece = self.agents.index(self.agent_selection) + 1
        next_agent_index = piece % 2
        opp_piece = next_agent_index + 1
        sum_own = len([i for i, value in enumerate(self.board) if i % self.board_width == column and value == piece])
        sum_opp = len([i for i, value in enumerate(self.board) if i % self.board_width == column and value == opp_piece])
        if sum_own > sum_opp:
            return self.agent_selection
        if sum_opp > sum_own:
            return self.agents[next_agent_index]
        return None

    def check_for_winner(self):
        """Returns True if the current agent has won the game, False otherwise."""
        board = np.array(self.board).reshape(self.board_height, self.board_width)
        piece = self.agents.index(self.agent_selection) + 1

        # Check horizontal locations for win
        column_count = self.board_width
        row_count = self.board_height

        for c in range(column_count - 3):
            for r in range(row_count):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(column_count):
            for r in range(row_count - 3):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(column_count - 3):
            for r in range(row_count - 3):
                if (
                    board[r][c] == piece
                    and board[r + 1][c + 1] == piece
                    and board[r + 2][c + 2] == piece
                    and board[r + 3][c + 3] == piece
                ):
                    return True

        # Check negatively sloped diagonals
        for c in range(column_count - 3):
            for r in range(3, row_count):
                if (
                    board[r][c] == piece
                    and board[r - 1][c + 1] == piece
                    and board[r - 2][c + 2] == piece
                    and board[r - 3][c + 3] == piece
                ):
                    return True

        return False
