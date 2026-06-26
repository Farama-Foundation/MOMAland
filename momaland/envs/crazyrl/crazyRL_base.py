"""The Base environment inheriting from pettingZoo Parallel environment class."""

from copy import copy
from typing import Optional
from typing_extensions import override

import numpy as np
import numpy.typing as npt
import pygame
from gymnasium import spaces
from pygame import DOUBLEBUF, OPENGL

from momaland.utils.env import MOParallelEnv


def _distance_to_target(agent_location: npt.NDArray[np.float32], target_location: npt.NDArray[np.float32]) -> float:
    return np.linalg.norm(agent_location - target_location)


CLOSENESS_THRESHOLD = 0.1
FPS = 20


class CrazyRLBaseParallelEnv(MOParallelEnv):
    """The Base environment inheriting from pettingZoo Parallel environment class.

    The main API methods of this class are:
    - step
    - reset
    - render
    - close
    - state

    they are defined in this main environment, as well as the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_space: The Space object corresponding to valid rewards
    """

    metadata = {
        "render_modes": ["human"],
        "is_parallelizable": True,
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        size: int = 3,
        num_drones: int = 4,
        init_flying_pos=np.array([[0, 0, 1], [1, 1, 1], [0, 1, 1], [2, 2, 1]]),
        init_target_location=np.array([1, 1, 2.5]),
    ):
        """Initialization of a CrazyRL environment.

        Args:
            render_mode (str, optional): The mode to display the rendering of the environment. Can be human or None.
            size (int, optional): Size of the area sides
            num_drones: amount of drones
            init_flying_pos: 2d array containing the coordinates of the agents
                is a (3)-shaped array containing the initial XYZ position of the drones.
            init_target_location: Array of the initial position of the moving target
        """
        self.num_drones = num_drones
        self.agents_names = np.array(["agent_" + str(i) for i in range(self.num_drones)])
        self.size = size  # The size of the square grid

        # locations
        self.init_flying_pos = {agent: init_flying_pos[i].copy() for i, agent in enumerate(self.agents_names)}
        self.agent_location = self.init_flying_pos.copy()
        self.previous_location = self.init_flying_pos.copy()  # for potential based reward

        # targets
        self.init_target_location = init_target_location.copy()
        self.target_location = init_target_location.copy()
        self.previous_target = init_target_location.copy()

        self.possible_agents = self.agents_names.tolist()
        self.timestep = 0
        self.agents = []
        self.size = size

        # spaces
        self.action_spaces = dict(zip(self.agents_names, [self._action_space() for agent in self.agents_names]))
        self.observation_spaces = dict(zip(self.agents_names, [self._observation_space() for agent in self.agents_names]))
        self.reward_spaces = dict(zip(self.agents_names, [self._reward_space() for agent in self.agents_names]))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.window_size = 900  # The size of the PyGame window
            self.window = None
            self.clock = None

    def _observation_space(self):
        return spaces.Box(
            low=np.tile(np.array([-self.size, -self.size, 0], dtype=np.float32), self.num_drones + 1),
            high=np.tile(np.array([self.size, self.size, 3], dtype=np.float32), self.num_drones + 1),
            shape=(3 * (self.num_drones + 1),),  # coordinates of the drones and the target
            dtype=np.float32,
        )

    def _action_space(self):
        return spaces.Box(low=-1 * np.ones(3, dtype=np.float32), high=np.ones(3, dtype=np.float32), dtype=np.float32)

    def _reward_space(self):
        return spaces.Box(
            low=np.array([-10, -10], dtype=np.float32),
            high=np.array([1, np.inf], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def action_space(self, agent):
        """Returns the action space for the given agent."""
        return self.action_spaces[agent]

    def observation_space(self, agent):
        """Returns the observation space for the given agent."""
        return self.observation_spaces[agent]

    def reward_space(self, agent):
        """Returns the reward space for the given agent."""
        return self.reward_spaces[agent]

    def _transition_state(self, action):
        """Computes the action passed to `.step()` into action matching the mode environment. Must be implemented in a subclass.

        Args:
            action : ndarray | dict[..]. The input action for one drones
        """
        raise NotImplementedError

    def _compute_obs(self):
        obs = dict()

        for agent in self.agents_names:
            obs[agent] = self.agent_location[agent].copy()
            obs[agent] = np.append(obs[agent], self.target_location)

            for other_agent in self.agents_names:
                if other_agent != agent:
                    obs[agent] = np.append(obs[agent], self.agent_location[other_agent])
            obs[agent] = np.array(obs[agent], dtype=(np.float32))

        return obs

    def _compute_reward(self):
        reward = dict()

        for agent in self.agents_names:
            reward_far_from_other_agents = 0
            reward_close_to_target = 0

            # mean distance to the other agents
            for other_agent in self.agents_names:
                if other_agent != agent:
                    reward_far_from_other_agents += np.linalg.norm(
                        self.agent_location[agent] - self.agent_location[other_agent]
                    )

            reward_far_from_other_agents /= self.num_drones - 1

            # distance to the target
            # (!) targets and locations must be updated before this
            dist_from_old_target = _distance_to_target(self.agent_location[agent], self.previous_target)
            old_dist = _distance_to_target(self.previous_location[agent], self.previous_target)

            # reward should be new_potential - old_potential but since the distances should be negated we reversed the signs
            # -new_potential - (-old_potential) = old_potential - new_potential
            reward_close_to_target = old_dist - dist_from_old_target

            # collision between two drones
            for other_agent in self.agents_names:
                if other_agent != agent and (
                    np.linalg.norm(self.agent_location[agent] - self.agent_location[other_agent]) < CLOSENESS_THRESHOLD
                ):
                    reward_far_from_other_agents = -10
                    reward_close_to_target = -10

            # collision with the ground or the target
            if (
                self.agent_location[agent][2] < CLOSENESS_THRESHOLD
                or np.linalg.norm(self.agent_location[agent] - self.target_location) < CLOSENESS_THRESHOLD
            ):
                reward_far_from_other_agents = -10
                reward_close_to_target = -10

            reward[agent] = np.array([reward_close_to_target, reward_far_from_other_agents], dtype=np.float32)

        return reward

    def _compute_terminated(self):
        terminated = dict()

        for agent in self.agents:
            terminated[agent] = False

        for agent in self.agents:
            # collision between two drones
            for other_agent in self.agents:
                if other_agent != agent:
                    terminated[agent] = terminated[agent] or (
                        np.linalg.norm(self.agent_location[agent] - self.agent_location[other_agent]) < CLOSENESS_THRESHOLD
                    )

            # collision with the ground
            terminated[agent] = terminated[agent] or (self.agent_location[agent][2] < CLOSENESS_THRESHOLD)

            # collision with the target
            terminated[agent] = terminated[agent] or (
                np.linalg.norm(self.agent_location[agent] - self.target_location) < CLOSENESS_THRESHOLD
            )

            if terminated[agent]:
                for other_agent in self.agents:
                    terminated[other_agent] = True
                self.agents = []

            terminated[agent] = bool(terminated[agent])

        return terminated

    def _compute_truncation(self):
        if self.timestep == 200:
            truncation = {agent: True for agent in self.agents_names}
            self.agents = []
            self.timestep = 0
        else:
            truncation = {agent: False for agent in self.agents_names}
        return truncation

    def _compute_info(self):
        info = dict()
        for agent in self.agents_names:
            info[agent] = {}
        return info

    # PettingZoo API
    @override
    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0
        self.agents = copy(self.possible_agents)
        self.target_location = self.init_target_location.copy()
        self.previous_target = self.init_target_location.copy()

        self.agent_location = self.init_flying_pos.copy()
        self.previous_location = self.init_flying_pos.copy()

        observation = self._compute_obs()
        infos = self._compute_info()

        if self.render_mode == "human":
            self.render()
        return observation, infos

    @override
    def step(self, actions):
        self.timestep += 1

        new_locations = self._transition_state(actions)
        self.previous_location = self.agent_location
        self.agent_location = new_locations

        if self.render_mode == "human":
            self.render()

        observations = self._compute_obs()
        rewards = self._compute_reward()
        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos

    @override
    def render(self):
        """Renders the current frame of the environment. Only works in human rendering mode."""
        from OpenGL.GL import (
            GL_AMBIENT,
            GL_AMBIENT_AND_DIFFUSE,
            GL_BLEND,
            GL_COLOR_BUFFER_BIT,
            GL_COLOR_MATERIAL,
            GL_DEPTH_BUFFER_BIT,
            GL_DEPTH_TEST,
            GL_DIFFUSE,
            GL_FRONT_AND_BACK,
            GL_LIGHT0,
            GL_LIGHTING,
            GL_MODELVIEW,
            GL_MODELVIEW_MATRIX,
            GL_ONE_MINUS_SRC_ALPHA,
            GL_POSITION,
            GL_PROJECTION,
            GL_SMOOTH,
            GL_SRC_ALPHA,
            glBlendFunc,
            glClear,
            glColor4f,
            glColorMaterial,
            glEnable,
            glGetFloatv,
            glLight,
            glLightfv,
            glLineWidth,
            glLoadIdentity,
            glMatrixMode,
            glMultMatrixf,
            glPopMatrix,
            glPushMatrix,
            glShadeModel,
        )
        from OpenGL.raw.GLU import gluLookAt, gluPerspective

        from momaland.envs.crazyrl.gl_utils import axes, field, point, target_point

        def init_window():
            """Initializes the PyGame window."""
            # pygame.init() also starts the unused audio mixer and joystick subsystems, whose device
            # enumeration adds ~0.4s of startup (see Farama-Foundation/MOMAland#71); only the display
            # subsystem (initialized below) is needed.
            pygame.display.init()
            pygame.display.set_caption("Crazy RL")

            self.window = pygame.display.set_mode((self.window_size, self.window_size), DOUBLEBUF | OPENGL)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glLineWidth(1.5)

            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

            glMatrixMode(GL_PROJECTION)
            gluPerspective(75, (self.window_size / self.window_size), 0.1, 50.0)

            glMatrixMode(GL_MODELVIEW)
            gluLookAt(3, -11, 3, 0, 0, 0, 0, 0, 1)

            self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
            glLoadIdentity()

        if self.window is None:
            init_window()

        # if self.clock is None and self.render_mode == "human":
        self.clock = pygame.time.Clock()

        glLoadIdentity()

        # init the view matrix
        glPushMatrix()
        glLoadIdentity()

        # multiply the current matrix by the get the new view matrix and store the final view matrix
        glMultMatrixf(self.viewMatrix)
        self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

        # apply view matrix
        glPopMatrix()
        glMultMatrixf(self.viewMatrix)

        glLight(GL_LIGHT0, GL_POSITION, (-1, -1, 5, 1))  # point light from the left, top, front

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for agent in self.agent_location.values():
            glPushMatrix()
            point(np.array([agent[0], agent[1], agent[2]]))

            glPopMatrix()

        glColor4f(0.5, 0.5, 0.5, 1)
        field(self.size)
        axes()

        # for target in self.target_location:
        glPushMatrix()
        target_point(np.array([self.target_location[0], self.target_location[1], self.target_location[2]]))
        glPopMatrix()

        pygame.event.pump()
        pygame.display.flip()

    @override
    def state(self):
        states = tuple(self._compute_obs()[agent].astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    @override
    def close(self):
        if self.render_mode == "human":
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
