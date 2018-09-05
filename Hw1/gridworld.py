"""
Example 3.5 (Grid world) from Sutton and Barto
"""

import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np


class GridWorld(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1
    }

    def __init__(self):

        # specific positions
        self.A_x = 2
        self.A_y = 5
        self.prime_A_x = 2
        self.prime_A_y = 1
        self.B_x = 4
        self.B_y = 5
        self.prime_B_x = 4
        self.prime_B_y = 3

        # boundary of grid
        self.x_min = 0
        self.x_max = 5
        self.y_min = 0
        self.y_max = 5

        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Discrete(5)

        self.viewer = None
        self.agent_trans = None
        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, y = state
        x_old, y_old = state

        done = False
        boost_mode = False

        if (x_old == self.A_x and y_old == self.A_y) or (x_old == self.B_x and y_old == self.B_y):
            boost_mode = True

        if boost_mode:
            if x_old == self.A_x and y_old == self.A_y:
                x = self.prime_A_x
                y = self.prime_A_y
                reward = 10

            if x_old == self.B_x and y_old == self.B_y:
                x = self.prime_B_x
                y = self.prime_B_y
                reward = 5

        else:
            if action == 0:
                x = x_old
                y = y_old + 1
                reward = 0.0
            elif action == 1:
                x = x_old + 1
                y = y_old
                reward = 0.0
            elif action == 2:
                x = x_old
                y = y_old - 1
                reward = 0.0
            elif action == 3:
                x = x_old - 1
                y = y_old
                reward = 0.0

            if x > 5 or x < 0 or y > 5 or y < 0:
                x = x_old
                y = y_old
                reward = -1

        self.state = (x, y)

        return np.array(self.state), reward, done

    def reset(self):
        self.state = (env.state_space.sample()+1, env.state_space.sample()+1)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500
        scale = 100
        state = self.state
        x, y = state

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(6):
                line_s_x = i * scale
                line_s_y = 0 * scale
                line_e_x = i * scale
                line_e_y = 5 * scale
                line = self.viewer.draw_line((line_s_x, line_s_y), (line_e_x, line_e_y))
                self.viewer.add_geom(line)
            for i in range(6):
                line_s_y = i * scale
                line_s_x = 0 * scale
                line_e_y = i * scale
                line_e_x = 5 * scale
                line = self.viewer.draw_line((line_s_x, line_s_y), (line_e_x, line_e_y))
                self.viewer.add_geom(line)
            agent_size = 30
            left, right, top, bottom = -agent_size, +agent_size, +agent_size, -agent_size
            agent = rendering.FilledPolygon([(left, bottom), (left, top), (right, top), (right, bottom)])
            self.agent_trans = rendering.Transform()
            agent.add_attr(self.agent_trans)
            agent.set_color(.8, .6, .4)
            self.viewer.add_geom(agent)

        if self.state is None:
            return None

        self.agent_trans.set_translation((x-0.5)*100, (y-0.5)*100)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


env = GridWorld()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
