"""MO-Breakthrough.

|--------------------|--------------------------------------------------|
| Actions            | Discrete                                         |
| Parallel API       | No                                               |
| Manual Control     | No                                               |
| Agents             | 2                                                |
| Action Shape       | (1,)                                             |
| Action Values      | Discrete(board_width=8 * board_height=8 * 3)     |
| Observation Shape  | (board_height=8, board_width=8, 2)               |
| Observation Values | [0,1]                                            |
| Reward Shape       | (num_objectives=4,)                              |
"""

from __future__ import annotations

from typing_extensions import override

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.logger import warn
from gymnasium.utils import EzPickle
from pettingzoo.utils import AgentSelector, wrappers

from momaland.utils.env import MOAECEnv


def env(**kwargs):
    """Returns the wrapped MOBreakthrough environment in `AEC` format.

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
    return MOBreakthrough(**kwargs)


class MOBreakthrough(MOAECEnv, EzPickle):
    """Multi-objective Breakthrough.

    MO-Breakthrough is a multi-objective variant of the two-player, single-objective turn-based board game Breakthrough.
    In Breakthrough, players start with two rows of identical pieces in front of them, on an 8x8 board, and try to reach
    the opponent's home row with any piece. The first player to move a piece on their opponent's home row wins. Players
    move alternatingly, and each piece can move one square straight forward or diagonally forward. Opponent pieces can also
    be captured, but only by moving diagonally forward, not straight.
    MO-Breakthrough extends this game with up to three additional objectives: a second objective that incentivizes faster
    wins, a third one for capturing opponent pieces, and a fourth one for avoiding the capture of the agent's own pieces.
    Additionally, the board width can be modified from 3 to 20 squares, and the board height from 5 to 20 squares.

    ## Observation Space
    The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described
    below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section below.
    The main observation space is 2 planes of a board_height * board_width grid (a board_height * board_width * 2 tensor).
    Each plane represents a specific agent's pieces, and each location in the grid represents the placement of the
    corresponding agent's piece. 1 indicates that the agent has a piece placed in the given location, and 0 indicates they
    do not have a piece in that location (meaning that either the cell is empty, or the other agent has a piece in that
    location).

    ## Legal Actions Mask
    The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation.
    The `action_mask` is a binary vector where each index of the vector represents whether the represented action is legal
    or not; the action encoding is described in the Action Space section below.
    The `action_mask` will be all zeros for any agent except the one whose turn it is.

    ## Action Space
    The action space is the set of integers from 0 to board_width*board_height*3 (exclusive). If a piece at coordinates
    (x,y) is moved, this is encoded as the integer x * 3 * board_height + y * 3 + z where z == 0 for left diagonal, 1 for
    straight, and 2 for right diagonal move.

    ## Rewards
    Dimension 0: If an agent moves one of their pieces to the opponent's home row, they will be rewarded 1 point. At the
    same time, the opponent agent will be awarded -1 point. There are no draws in Breakthrough.
    Dimension 1: If an agent wins, they get a reward of 1-(move_count/max_moves) to incentivize faster wins. The losing
    opponent gets the negated reward. In case of a draw, both agents get 0.
    Dimension 2: (optional) The number of opponent pieces (divided by the max number of pieces) an agent has captured.
    Dimension 3: (optional) The negative number of pieces (divided by the max number of pieces)
    an agent has lost to the opponent.

    ## Starting State
    The starting board is empty except for the first two rows that are filled with pieces of player 0, and the last two rows that are filled with pieces of player 1.

    ## Arguments
    - 'board_width': The width of the board (from 3 to 20)
    - 'board_height': The height of the board (from 5 to 20)
    - 'num_objectives': The number of objectives (from 1 to 4)
    - 'render_mode': The render mode.

    ## Version History
    """

    metadata = {
        "render_modes": ["ansi", "human", "rgb_array"],
        "name": "mobreakthrough_v0",
        "is_parallelizable": False,
        "render_fps": 4,
    }

    OFF_BOARD = -1
    ANGLES = ["LEFT", "STRAIGHT", "RIGHT"]

    def __init__(
        self,
        board_width: int = 8,
        board_height: int = 8,
        num_objectives: int = 4,
        render_mode: str | None = None,
    ):
        """Initializes a new MOBreakthrough environment.

        Args:
            board_width: The width of the board (from 3 to 20)
            board_height: The height of the board (from 5 to 20)
            num_objectives: The number of objectives (from 1 to 4)
            render_mode: The render mode.
        """
        EzPickle.__init__(
            self,
            board_width,
            board_height,
            num_objectives,
            render_mode,
        )
        if not (3 <= board_width <= 20):
            raise ValueError("Config parameter board_width must be between 3 and 20.")

        elif not (5 <= board_height <= 20):
            raise ValueError("Config parameter board_height must be between 5 and 20.")

        elif not (1 <= num_objectives <= 4):
            raise ValueError("Config parameter num_objectives must be between 1 and 4.")

        self.board_width = board_width
        self.board_height = board_height
        self.num_objectives = num_objectives
        self.board_size = board_height * board_width
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.move_count = 0
        self.max_turns = 4 * board_width * (board_height - 3) + 1
        self.max_move = board_height * board_width * 3
        self._cumulative_rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        self._initialize_board(board_height, board_width)
        self.legal_moves = self._legal_moves()
        self.render_mode = render_mode

        self.action_spaces = {agent: spaces.Discrete(self.max_move) for agent in self.agents}
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0,
                        high=1,
                        shape=(board_height, board_width, len(self.agents)),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.MultiBinary(self.max_move),
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

        # pygame rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._cell = max(22, min(56, 520 // max(board_width, board_height)))
        self._header = 40
        self._panel_w = 210
        self.window_size = (board_width * self._cell + self._panel_w, board_height * self._cell + self._header)
        self.window = None
        self.clock = None

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
            self.print_board()
            return

        # human / rgb_array: draw the board as a checkerboard with colored pieces per player.
        cols, rows, cell = self.board_width, self.board_height, self._cell
        if self.window is None:
            # Lightweight init: only the subsystems used (display + font), and the display only on the
            # human path. pygame.init() would also start the unused audio/joystick subsystems (#71).
            pygame.font.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("MO-Breakthrough")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        font = pygame.font.SysFont("Arial", 14)
        small_font = pygame.font.SysFont("Arial", 12)

        light, dark = (236, 223, 200), (181, 148, 107)
        p_color = {1: (60, 110, 205), 2: (210, 70, 70)}  # player_0, player_1
        header = self._header

        self.window.fill((247, 245, 238))
        self.window.blit(title_font.render("MO-Breakthrough", True, (20, 20, 20)), (10, 10))

        radius = int(cell * 0.36)
        for col in range(cols):
            for row in range(rows):
                # Draw row 0 at the bottom so player_0 (which starts low and advances upward) is at the bottom.
                sx = col * cell
                sy = header + (rows - 1 - row) * cell
                pygame.draw.rect(self.window, light if (col + row) % 2 == 0 else dark, pygame.Rect(sx, sy, cell, cell))
                piece = int(self.board[col][row])
                if piece != 0:
                    center = (sx + cell // 2, sy + cell // 2)
                    pygame.draw.circle(self.window, p_color[piece], center, radius)
                    pygame.draw.circle(self.window, (30, 30, 30), center, radius, 2)

        # Side panel: turn / winner, move count, pieces left, objectives.
        panel_x = cols * cell + 16
        p0_left = int((self.board == 1).sum())
        p1_left = int((self.board == 2).sum())
        game_over = any(self.terminations.values())

        y = header + 6
        if game_over:
            # The winner is whoever still has pieces / reached the far row; infer it from the board.
            if 1 in self.board[:, rows - 1] or p1_left == 0:
                winner = "player_0"
            else:
                winner = "player_1"
            self.window.blit(title_font.render(f"{winner} wins!", True, (20, 20, 20)), (panel_x, y))
        else:
            self.window.blit(font.render("to move:", True, (60, 60, 60)), (panel_x, y))
            pidx = self.possible_agents.index(self.agent_selection)
            pygame.draw.circle(self.window, p_color[pidx + 1], (panel_x + 80, y + 8), 8)
            self.window.blit(font.render(self.agent_selection, True, (20, 20, 20)), (panel_x + 96, y))
        y += 34

        self.window.blit(font.render(f"move: {self.move_count}", True, (40, 40, 40)), (panel_x, y))
        y += 26
        for label, piece, left in [("player_0", 1, p0_left), ("player_1", 2, p1_left)]:
            pygame.draw.circle(self.window, p_color[piece], (panel_x + 8, y + 8), 7)
            self.window.blit(small_font.render(f"{label}: {left} pieces", True, (20, 20, 20)), (panel_x + 22, y))
            y += 22
        y += 8
        self.window.blit(small_font.render(f"objectives: {self.num_objectives}", True, (90, 90, 90)), (panel_x, y))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    @override
    def observe(self, agent):  # currently using a fixed layer for the current player, instead of fixed layers by color
        cur_player = self.possible_agents.index(agent)
        opp_player = (cur_player + 1) % 2
        cur_p_board = np.equal(self.board, cur_player + 1)
        opp_p_board = np.equal(self.board, opp_player + 1)
        observation = np.stack([cur_p_board, opp_p_board], axis=2).astype(np.int8)
        actions = self.legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros(self.max_move, "int8")
        action_mask[list(actions)] = 1
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
        legal_moves = set()
        cur_player = self.possible_agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2
        cur_piece = cur_player + 1
        opp_piece = opp_player + 1
        move_direction = 1 if cur_piece == 1 else -1
        for index, value in np.ndenumerate(self.board):
            x, y = index
            if value == cur_piece:
                if self._get_square(x, y + move_direction) == 0:  # move straight ahead possible
                    legal_moves.add(self._move_to_int(x, y, "STRAIGHT"))
                if x < self.board_width - 1:
                    if (
                        self._get_square(x + 1, y + move_direction) == 0
                        or self._get_square(x + 1, y + move_direction) == opp_piece
                    ):  # move diagonally to the right possible
                        legal_moves.add(self._move_to_int(x, y, "RIGHT"))
                if x > 0:
                    if (
                        self._get_square(x - 1, y + move_direction) == 0
                        or self._get_square(x + 1, y + move_direction) == opp_piece
                    ):  # move diagonally to the left possible
                        legal_moves.add(self._move_to_int(x, y, "LEFT"))
        return legal_moves

    def _get_square(self, x, y):
        """Returns the piece at the given coordinates, provided the coordinates are legal. Otherwise returns MOBreakthrough.OFF_BOARD."""
        if x < 0 or x > self.board_width - 1 or y < 0 or y > self.board_height - 1:
            return MOBreakthrough.OFF_BOARD
        return self.board[x][y]

    @override
    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)

        # assert valid move
        assert action in self.legal_moves, "played illegal move."

        # make the move
        x, y, direction = self._int_to_move(action)
        agent = self.agent_selection
        agent_index = self.possible_agents.index(agent)
        next_agent = self.possible_agents[(agent_index + 1) % 2]
        agent_piece = agent_index + 1
        move_direction = 1 if agent_piece == 1 else -1
        self.board[x][y] = 0
        capture = False
        if self.board[x + MOBreakthrough.ANGLES.index(direction) - 1][y + move_direction] != 0:
            capture = True
        self.board[x + MOBreakthrough.ANGLES.index(direction) - 1][y + move_direction] = agent_piece
        self.move_count += 1

        # handle the rewards
        self.rewards = {agent: np.zeros(self.num_objectives) for agent in self.agents}
        if capture:
            if self.num_objectives > 2:
                self.rewards[agent][2] = 1 / (self.board_width * 2)
            if self.num_objectives > 3:
                self.rewards[next_agent][3] = -1 / (self.board_width * 2)
        if self.check_for_winner():
            self.rewards[agent][0] = 1
            self.rewards[next_agent][0] = -1
            if self.num_objectives > 1:
                self.rewards[agent][1] = 1 - (self.move_count / self.max_turns)
                self.rewards[next_agent][1] = -(1 - (self.move_count / self.max_turns))
            self.terminations = {agent: True for agent in self.agents}
        self._cumulative_rewards[agent] = np.zeros(self.num_objectives, dtype=np.float32)
        self._accumulate_rewards()

        # select the next agent
        self.agent_selection = self._agent_selector.next()
        self.legal_moves = self._legal_moves()
        if self.render_mode == "human":
            self.render()

    @override
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: np.zeros(self.num_objectives, dtype=np.float32) for agent in self.agents}
        self._cumulative_rewards = {agent: np.zeros(self.num_objectives, dtype=np.float32) for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.move_count = 0
        self._initialize_board(self.board_height, self.board_width)
        self.legal_moves = self._legal_moves()

    @override
    def close(self):
        """Close should release any graphical displays, subprocesses, network connections or any other environment
        data which should not be kept around after the user is no longer using the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def check_for_winner(self):
        """Checks if there is a winner and the game is over."""
        cur_player = self.possible_agents.index(self.agent_selection)
        opp_player = (cur_player + 1) % 2
        cur_piece = cur_player + 1
        opp_piece = opp_player + 1
        if not (opp_piece in self.board):
            return True
        if cur_piece == 1:
            home_row = self.board_height - 1
        else:
            home_row = 0
        if cur_piece in self.board[:, home_row]:
            return True
        return False

    def _initialize_board(self, board_height, board_width):
        """Initializes the board."""
        self.board = np.zeros((board_width, board_height))
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        piece = self.agents.index(self.agent_selection) + 1
        next_agent_index = piece % 2
        opp_piece = next_agent_index + 1
        self.board[:, :2] = piece
        self.board[:, -2:] = opp_piece

    def print_board(self):
        """Prints the board in a human-readable format."""
        rotated_board = self.board
        for row in range(self.board_height):
            for col in range(self.board_width):
                if rotated_board[col][row] == 0:
                    print(".  ", end="")
                else:
                    print(int(rotated_board[col][row]), " ", end="")
            print()
        print()

    def _move_to_int(self, x, y, direction):
        """Converts move coordinates and direction to integer move encoding."""
        return x * 3 * self.board_height + y * 3 + MOBreakthrough.ANGLES.index(direction)

    def _int_to_move(self, move):
        """Converts integer move encoding to move coordinates and direction."""
        return (
            (move // 3) // self.board_height,
            (move // 3) % self.board_height,
            MOBreakthrough.ANGLES[move % 3],
        )
