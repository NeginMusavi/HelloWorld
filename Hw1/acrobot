
"""classic Acrobot task"""
from gym import core, spaces
from gym.utils import seeding
import numpy as np
from numpy import sin, cos, pi


class AcrobotEnv(core.Env):

    dt = 0.1
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    I1 = 1.0
    I2 = 1.0

    delta = np.pi / 4

    dq1_max = 4 * np.pi
    dq2_max = 9 * np.pi

    torque_options = [-1.0, 0.0, +1.0]

    q1_min = np.pi / 2 - delta
    q1_max = np.pi / 2 + delta

    q2_min = 0 - delta
    q2_max = 0 + delta

    def __init__(self):
        self.viewer = None
        high = np.array([1.0, 1.0, 1.0, 1.0, self.dq1_max, self.dq2_max])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(3)
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        s = self.state
        return np.array([cos(s[0]), np.sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])

    def step(self, a):
        old_state = self.state
        torque = self.torque_options[a]

        s_augmented = np.append(old_state, torque)

        new_state = rk4(self._dsdt, s_augmented, [0, self.dt])
        new_state = new_state[-1]
        new_state = new_state[:4]

        if new_state[0] > 2 * pi:
            new_state[0] = new_state[0] - 2 * pi
        if new_state[0] < 2 * pi:
            new_state[0] = new_state[0] + 2 * pi

        if new_state[1] > 2 * pi:
            new_state[1] = new_state[1] - 2 * pi
        if new_state[1] < 2 * pi:
            new_state[1] = new_state[1] + 2 * pi

        self.state = new_state
        if self.q1_min < new_state[0] < self.q1_max and self.q2_min < new_state[1] < self.q2_max:
            immediate_reward = +1
        else:
            immediate_reward = 0

        observations = np.array([cos(new_state[0]), np.sin(new_state[0]), cos(new_state[1]), sin(new_state[1]), new_state[2], new_state[3]])
        return observations, immediate_reward, {}

    def _dsdt(self, s_augmented, t):
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = l1/2
        lc2 = l1/2
        I1 = self.I1
        I2 = self.I2
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        q1 = s[0]
        q2 = s[1]
        dq1 = s[2]
        dq2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(q2)) + I2
        phi2 = m2 * lc2 * g * np.cos(q1 + q2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dq2 ** 2 * np.sin(q2) \
               - 2 * m2 * l1 * lc2 * dq2 * dq1 * np.sin(q2)  \
            + (m1 * lc1 + m2 * l1) * g * np.cos(q1 - np.pi / 2) + phi2

        ddq2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dq1 ** 2 * np.sin(q2) - phi2) \
                / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddq1 = -(d2 * ddq2 + phi1) / d1
        return dq1, dq2, ddq1, ddq2, 0.

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

        if s is None:
            return None

        p1 = [-self.l1 * np.cos(s[0]), self.l1 * np.sin(s[0])]

        p2 = [p1[0] - self.l2 * np.cos(s[0] + s[1]), p1[1] + self.l2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]

        for ((x, y), th) in zip(xys, thetas):
            l, r, t, b = 0, 1, .1, -.1
            trans = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(trans)
            link.set_color(1.0, 0.5, 0.5)
            joint = self.viewer.draw_circle(.1)
            joint.set_color(0.5, 0.5, 1.0)
            joint.add_attr(trans)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


def rk4(derivative, y0, t, *args, **kwargs):

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivative(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivative(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivative(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivative(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


env = AcrobotEnv()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, info = env.step(action)
        print(reward)
