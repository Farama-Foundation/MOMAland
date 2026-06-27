"""Multi-Objective Route Choice Game.

From Ramos, G. D. O., Radulescu, R., Nowe, A., & Tavares, A. R. (2020). Toll-based learning for minimising route_choice under heterogeneous preferences.
"""

import functools
import json
import os
import random
from collections import defaultdict
from typing_extensions import override

import networkx as nx
import numpy as np
import pygame
from gymnasium.logger import warn
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import EzPickle
from pettingzoo.utils import wrappers
from sympy import diff, lambdify, sympify

from momaland.utils.conversions import mo_parallel_to_aec
from momaland.utils.env import MOParallelEnv


def _flow_color(ratio):
    """Maps a congestion ratio in [0, 1] to an RGB color, green (free) -> yellow -> red (congested)."""
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio < 0.5:
        t = ratio / 0.5
        return (int(60 + t * 180), int(180 + t * 30), int(75 - t * 25))
    t = (ratio - 0.5) / 0.5
    return (int(240 - t * 30), int(210 - t * 160), int(50))


def parallel_env(**kwargs):
    """Env factory function for the route choice game."""
    return raw_env(**kwargs)


def env(**kwargs):
    """Auto-wrapper for the route choice game.

    Args:
        **kwargs: keyword args to forward to the parallel_env function.

    Returns:
        A fully wrapped AEC env
    """
    env = parallel_env(**kwargs)
    # convert parallel version of the env to an AEC version
    env = mo_parallel_to_aec(env)

    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    return env


def raw_env(**kwargs):
    """Env factory function for the route choice game."""
    return MORouteChoice(**kwargs)


class MORouteChoice(MOParallelEnv, EzPickle):
    """A `Parallel` environment where drivers learn to travel from a source to a destination while avoiding congestion.

    Multi-objective version of Braess' Paradox where drivers have two objectives: travel time and monetary cost.
    The environment is a road network and the agents are the drivers that needs to travel from an origin to a destination point.

    ## Observation Space
    This environment is stateless, so the observation space is a constant 0. (Discrete with shape (1,)).

    ## Action Space
    The action space is a discrete space representing the possible routes that the agent can take.
    The number of routes is different for each agent, as it depends on the number of possible routes for the OD pair of the agent.
    Selecting an action corresponds to choosing a route.

    ## Reward Space
    The reward space is a 2D vector containing rewards for:
    - Minimizing travel time (latency).
    - Minimizing monetary cost.

    ## Starting State
    The environment is stateless, so there is no starting state.

    ## Episode Termination
    The environment is stateless, so there are no episodes. Each "episode" is therefore terminated after each timestep.

    ## Episode Truncation
    Episodes are not truncated as there are terminated after each timestep.

    ## Arguments
    - `render_mode (str, optional)`: The mode to display the rendering of the environment. Can be human or None.
    - `problem_name (str, optional)`: The name of the road network that will be used.
    - `num_agents (int, optional)`: The number of drivers in the network.
    - `toll_mode (str, optional)`: The tolling mode that is used, tolls are either placed randomly "random" or using marginal cost tolling "mct".
    - `random_toll_percentage (float, optional)`: In the case of random tolling the percentage of roads that will be taxed.
    - `num_timesteps (int, optional)`: The number of timesteps (stateless, therefore always 1 timestep).

    ## Credits
    The code was adapted from [codebase of "Toll-Based Learning for Minimising Congestion under Heterogeneous Preferences"](https://github.com/goramos/marl-route-choice).
    """

    def __init__(
        self,
        problem_name="Braess_1_4200_10_c1",
        num_agents=4200,
        toll_mode="mct",
        random_toll_percentage=0.1,
        num_timesteps=1,
        render_mode=None,
    ):
        """Initializes the route choice game.

        Args:
            problem_name: the name of the network that will be used
            num_agents: number of agents in the network
            toll_mode: the tolling mode that is used, tolls are either placed randomly "random" or using marginal cost tolling "mct"
            random_toll_percentage: in the case of random tolling the percentage of roads that will be taxed
            num_timesteps: number of timesteps (stateless, therefore always 1 timestep)
            render_mode: render mode
        """
        EzPickle.__init__(
            self,
            problem_name,
            num_agents,
            toll_mode,
            random_toll_percentage,
            num_timesteps,
            render_mode,
        )
        # Read in the problem from the corresponding .json file in the networks directory
        self.graph, self.od, self.routes, self._max_route_length = self._read_problem(problem_name)
        # Keep track of the current flow on each link the network
        self.flows = {f"{edge[0]}-{edge[1]}": 0 for edge in self.graph.edges}
        self.avg_tt = 0.0

        # Episodes/Timesteps
        self.num_timesteps = num_timesteps
        self.episode_num = 0

        self.render_mode = render_mode
        self.possible_agents = ["agent_" + str(i) for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        # each driver gets assigned a random origin-destination (OD) pair by _init_state()
        self.drivers_od = self._init_state()
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # compute the possible routes each agent can take based on its OD pair
        route_choices_per_agent = [Discrete(len(self.routes[od])) for od in self.drivers_od]
        # action space can be different for agents when there are multiple OD pairs with different numbers of routes
        self.action_spaces = dict(zip(self.agents, route_choices_per_agent))
        # stateless setting, agents receive a constant '0' as an observation in each timestep
        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    Discrete(
                        1,
                    )
                ]
                * num_agents,
            )
        )
        # keep track of the maximum link latency and cost to scale the rewards returned to the agents
        self._max_link_latency = None
        self._max_link_cost = None
        # the latency and cost of links are scaled (at most 1), the maximum latency and cost is therefore at most the length of the longest route
        # latency and cost are both negative rewards, agents aim to find routes with minimal latency and cost
        self.reward_spaces = dict(zip(self.agents, [Box(low=-self._max_route_length, high=0, shape=(2,))] * num_agents))

        # Each arc can have a different latency function (e.g. constant travel time / travel time dependent on flow)
        self.latency_functions = dict()
        self.toll_mode = toll_mode  # "random" or "mct"
        assert (
            self.toll_mode == "random" or self.toll_mode == "mct"
        ), "chosen toll mode not supported, use either random or mct"
        self.random_toll_percentage = random_toll_percentage
        # Marginal cost is given by the product of the flow and the derivative of the latency function of the arc
        self.cost_function = dict()
        self._create_latency_and_cost_function(nx.get_edge_attributes(self.graph, "latency_function"), num_agents)

        # pygame rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = (760, 560)
        self.window = None
        self.clock = None
        self._node_pos = None  # pixel positions of the network nodes, computed lazily on first render

    metadata = {"render_modes": ["human", "rgb_array"], "name": "moroute_choice_v0", "render_fps": 2}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected
    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
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
        """Renders the road network and the current congestion on each link.

        Nodes are the junctions of the network; origin nodes are outlined in green and destination
        nodes in red (from the OD pairs). Each directed link is drawn as an arrow whose width and
        color encode its current flow (the number of drivers on it), from green (free flowing) to
        red (congested), making the cost of congestion - the core of Braess' paradox - visible. The
        link flow and the average travel time are printed on the frame.

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
                pygame.display.set_caption("MO-RouteChoice")
                self.window = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window = pygame.Surface(self.window_size)
            if self.clock is None:
                self.clock = pygame.time.Clock()
        if self._node_pos is None:
            self._node_pos = self._compute_node_positions()

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        font = pygame.font.SysFont("Arial", 13)

        # Origin and destination nodes, parsed from the "origin|destination" OD pairs.
        origins = {pair.split("|")[0] for pair in self.od}
        destinations = {pair.split("|")[1] for pair in self.od}

        max_flow = max([*self.flows.values(), 1])

        self.window.fill((245, 245, 240))
        self.window.blit(title_font.render("MO-RouteChoice  -  link congestion", True, (20, 20, 20)), (16, 12))
        self.window.blit(
            font.render(f"average travel time: {self.avg_tt:.3f}    drivers: {len(self.possible_agents)}", True, (60, 60, 60)),
            (16, 38),
        )

        node_radius = 20
        for u, v in self.graph.edges:
            flow = self.flows.get(f"{u}-{v}", 0)
            ratio = flow / max_flow
            start = np.array(self._node_pos[u], dtype=float)
            end = np.array(self._node_pos[v], dtype=float)
            direction = end - start
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
            # Stop the arrow at the node circles rather than at their centers.
            p0 = start + direction * node_radius
            p1 = end - direction * node_radius
            color = _flow_color(ratio)
            pygame.draw.line(self.window, color, p0, p1, 3 + int(ratio * 9))
            self._draw_arrowhead(p1, direction, color)
            mid = (p0 + p1) / 2
            self.window.blit(font.render(str(int(flow)), True, (20, 20, 20)), (mid[0] + 4, mid[1] - 18))

        for node, (x, y) in self._node_pos.items():
            if node in origins:
                ring_color, label = (40, 160, 60), "O"
            elif node in destinations:
                ring_color, label = (200, 50, 50), "D"
            else:
                ring_color, label = (120, 120, 120), ""
            pygame.draw.circle(self.window, (255, 255, 255), (int(x), int(y)), node_radius)
            pygame.draw.circle(self.window, ring_color, (int(x), int(y)), node_radius, 4)
            txt = font.render(f"{node}{(' ' + label) if label else ''}", True, (20, 20, 20))
            self.window.blit(txt, (int(x) - txt.get_width() // 2, int(y) - txt.get_height() // 2))

        # Congestion legend.
        for i, (lbl, ratio) in enumerate([("free", 0.0), ("busy", 0.5), ("congested", 1.0)]):
            ly = self.window_size[1] - 24
            lx = 16 + i * 130
            pygame.draw.line(self.window, _flow_color(ratio), (lx, ly), (lx + 28, ly), 3 + int(ratio * 9))
            self.window.blit(font.render(lbl, True, (60, 60, 60)), (lx + 34, ly - 8))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def _compute_node_positions(self):
        """Returns a dict mapping each node to its pixel position, using a deterministic spring layout."""
        layout = nx.spring_layout(self.graph, seed=42)
        xs = [p[0] for p in layout.values()]
        ys = [p[1] for p in layout.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        margin = 70
        width = self.window_size[0] - 2 * margin
        height = self.window_size[1] - 2 * margin - 30  # leave room for the title
        positions = {}
        for node, (x, y) in layout.items():
            nx_ = (x - min_x) / (max_x - min_x) if max_x > min_x else 0.5
            ny_ = (y - min_y) / (max_y - min_y) if max_y > min_y else 0.5
            positions[node] = (margin + nx_ * width, margin + 30 + ny_ * height)
        return positions

    def _draw_arrowhead(self, tip, direction, color):
        """Draws a small triangular arrowhead at `tip` pointing along `direction`."""
        perp = np.array([-direction[1], direction[0]])
        base = tip - direction * 12
        left = base + perp * 6
        right = base - perp * 6
        pygame.draw.polygon(self.window, color, [tip, left, right])

    @override
    def close(self):
        """Closes the rendering window."""
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
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # Reset the flows of each arc
        self.flows = {f"{edge[0]}-{edge[1]}": 0 for edge in self.graph.edges}
        observations = {agent: 0 for agent in self.agents}
        self.episode_num = 0

        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _init_state(self):
        """Initializes the state of the environment. This is called by reset()."""
        # randomly distribute drivers among possible OD pairs
        drivers_od = [random.choice(self.od) for _ in self.agents]
        return drivers_od

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

        # - Actions -#
        # keep track of the flow on each route, update the flow on the links of the roads at once afterward
        agents_on_routes = defaultdict(int)
        # keep track of each route selected by each agent
        agent_routes = []
        for i, agent in enumerate(self.agents):
            act = actions[agent]
            # get the OD pair of the agent, so we know which route the agent has chosen
            agent_od = self.drivers_od[i]
            agent_route = self.routes[agent_od][act]
            agents_on_routes[agent_route] += 1
            # save chosen route of agent
            agent_routes.append(agent_route)
        # add the flow of all routes to the links of each route:
        for route in agents_on_routes:
            self._add_flow_to_route(route, agents_on_routes[route])

        # - Observations -#
        # return constant observations '0' as this is a stateless setting
        observations = {agent: 0 for agent in self.agents}

        # - Infos -#
        # Infos contain the actual (unscaled) travel time and tolling of each agent
        infos = {agent: {} for agent in self.agents}

        # - Rewards -#
        # compute latency and cost of each route
        latency_routes, cost_routes, latency_routes_scaled, cost_routes_scaled = self._compute_latency_and_cost(
            agents_on_routes
        )
        # reset avg_tt
        self.avg_tt = 0.0

        rewards = dict()
        for i in range(len(self.agents)):
            # get the route which was taken by the agent
            agent_route = agent_routes[i]
            latency_reward = latency_routes_scaled[agent_route]
            cost_reward = cost_routes_scaled[agent_route]
            # retrieve the ID of the agent
            agent_id = self.agents[i]
            infos[agent_id] = {"latency": latency_routes[agent_route], "cost": cost_routes[agent_route]}
            self.avg_tt += latency_routes[agent_route]
            rewards[agent_id] = np.array([-latency_reward, -cost_reward], dtype=np.float32)
        # compute the average travel time of all agents
        self.avg_tt /= len(self.agents)

        # stateless bandit setting where each episode only lasts 1 timestep
        self.terminations = {agent: True for agent in self.agents}
        self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.truncations, self.terminations, infos

    # - Helper Methods -#
    def _read_problem(self, problem_name):
        """Reads in the .JSON file of the chosen problem.

        Parses the network which is loaded in NetworkX as well as the
        possible origin/destination (OD) pairs and the possible routes the agents can choose to travel from the origins
        to the destinations.

        Args:
            problem_name: the name of the problem which will be used, needs to correspond to the name of a .json file in the './networks/' directory.

        Returns: a tuple containing the following items in order:
            - graph: a NetworkX representation of the network
            - od: the possible origin/destination pairs in this problem
            - routes: the possible routes that can be used to travel from the origins to the destinations
            - max_route_length: the length of the longest route in the network
        """
        # if problem file already contains '.json' extension, ignore, else add extension to problem name
        if not problem_name.endswith(".json"):
            problem_name = problem_name + ".json"
        # open the .json file of the problem name from the local 'networks/' directory
        local_problem_file = os.path.join(os.path.dirname(__file__), "networks", problem_name)
        with open(local_problem_file) as graph_json:
            # load the .json
            data = json.load(graph_json)
            # - Graph -#
            graph = nx.node_link_graph(data["graph"], edges="links")
            # - Origin/Destination pairs -#
            od = data["od"]
            # - Possible routes for OD pairs -#
            routes = data["routes"]
            # compute max route length
            max_route_length = 0
            # routes is a dict with OD pairs as keys and lists of corresponding roads as values
            for od_key in routes:
                # routes contains a list of routes for each OD pair
                for route in routes[od_key]:
                    max_route_length = max(max_route_length, len(route.split(",")))

            return graph, od, routes, max_route_length

    def _create_latency_and_cost_function(self, edges_latency_attributes, num_agents):
        """Creates latency and cost functions for each edge of the network.

        Edges latency and cost functions are based on the .JSON file of the chosen problem. Each edge can have different travel times (latency function) and monetary costs.

        Args:
            edges_latency_attributes: parsed latency functions of links of the network
            num_agents: the total number of agents in the network
        """
        for edge in edges_latency_attributes:
            # retrieve latency attributes
            expr = edges_latency_attributes[edge]["expr"]
            param = edges_latency_attributes[edge]["param"]
            constants = edges_latency_attributes[edge]["constants"]
            # keys of latency_function and cost_function dict are "source-target" strings instead of tuples
            edge_name = f"{edge[0]}-{edge[1]}"
            # - Latency Function - #
            latency_formula = sympify(expr)
            simplified_latency = latency_formula.subs(constants)
            self.latency_functions[edge_name] = simplified_latency

            # - Cost Function - #
            # two toll modes are supported, either tolls are placed randomly on x% of roads OR marginal cost tolling is applied
            if self.toll_mode == "random":
                # if tolls are placed randomly, each road has a 'random_toll_percentage' chance of containing a toll equal to its latency
                if random.random() < self.random_toll_percentage:
                    self.cost_function[edge_name] = simplified_latency
                else:
                    self.cost_function[edge_name] = sympify("0")

            elif self.toll_mode == "mct":
                latency_deriv = diff(latency_formula, param)
                # marginal cost toll is computed as the product of the flow and the derivative of the latency function
                simplified_deriv = latency_deriv.subs(constants)
                mct_formula = sympify(f"{param}*" + str(simplified_deriv))
                self.cost_function[edge_name] = mct_formula

            # - Max Latency and Cost - #
            # keep track of max latency and cost to later scale latency and costs of links for rewards
            # check if this link produces maximum latency
            current_latency = lambdify(param, self.latency_functions[edge_name])(num_agents)
            if self._max_link_latency is None or self._max_link_latency < current_latency:
                self._max_link_latency = current_latency
            # check if this link produces maximum latency
            current_cost = lambdify(param, self.cost_function[edge_name])(num_agents)
            if self._max_link_cost is None or self._max_link_cost < current_cost:
                self._max_link_cost = current_cost

    def _add_flow_to_route(self, route, flow_to_add):
        """Adds 'flow_to_add' cars to all the links of 'route'.

        This is needed to compute the latency and cost after drivers have chosen their routes. Each driver on a road contributes 1 to the total flow on that road.

        Args:
            route: the route to which the flow should be added
            flow_to_add: the quantity of flow to add (the amount of drivers that chose this road)

        Returns:
            /
        """
        # Routes are strings which consists of links joined by ','
        links_of_route = route.split(",")
        # Add the flow to each link of the route
        for link in links_of_route:
            self.flows[link] += flow_to_add

    def _compute_latency_and_cost(self, routes):
        """Compute the latency and cost of a specific route based on the flow on its links.

        The latency and cost of a route is the sum of the latencies and costs of its links.

        Args:
            routes: the routes for which the total latency and cost will be computed. Contains all roads that
            were used by at least 1 agent.

        Returns:
            total_latencies: the total latency of each provided route
            total_cost: the total (monetary) cost of each provided route
        """
        all_used_routes = list(routes.keys())
        total_latencies = {route: 0 for route in all_used_routes}
        total_latencies_scaled = {route: 0 for route in all_used_routes}
        total_cost = {route: 0 for route in all_used_routes}
        total_cost_scaled = {route: 0 for route in all_used_routes}
        for route in routes:
            # get links of route
            links_of_route = route.split(",")
            # compute the total latency and cost of this specific route
            total_latencies[route] = sum(map(lambda link: self._get_link_latency(link), links_of_route))
            total_latencies_scaled[route] = (
                sum(map(lambda link: self._get_link_latency(link) / self._max_link_latency, links_of_route))
                / self._max_route_length
            )
            total_cost[route] = sum(map(lambda link: self._get_link_cost(link) / self._max_link_cost, links_of_route))
            total_cost_scaled[route] = (
                sum(map(lambda link: self._get_link_cost(link) / self._max_link_cost, links_of_route)) / self._max_route_length
            )
        # return the total (sum) latency and cost of this specific route
        return total_latencies, total_cost, total_latencies_scaled, total_cost_scaled

    def _get_link_latency(self, link):
        """Computes the latency of a link in the network.

        The latency of a link is its current latency (based on its flow)

        Args:
            link: the link for which the latency is computed

        Returns:
            link_latency: the latency of the link
        """
        latency_c = lambdify("f", self.latency_functions[link])
        return latency_c(self.flows[link])

    def _get_link_cost(self, link):
        """Computes the monetary cost of a link in the network.

        The monetary cost of a link is its current cost (based on its flow)

        Args:
            link: the link for which the cost is computed

        Returns:
            link_cost: the cost of the link
        """
        cost_c = lambdify("f", self.cost_function[link])
        return cost_c(self.flows[link])
