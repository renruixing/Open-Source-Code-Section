import numpy as np


class OU_Noise():
    def __init__(self, index, action_shape, mu=0, theta=0.15, std=0.25, dt=1e-2, scale=1, bound=1, decay=0.999):
        self.class_name = "OU"
        self.index = index
        self.action_shape = action_shape
        # 噪声均值
        self.mu = mu
        # 噪声回归速率
        self.theta = theta
        # 噪声扰动程度
        self.std = std
        self.dt = dt
        # 噪声尺度
        self.scale = scale
        # 噪声边界值
        self.bound = bound
        # 噪声衰减
        self.decay = decay
        self.reset()

    def reset(self):
        self.state = self.mu * np.ones(shape=self.action_shape)

    def get_noise(self):
        x = self.state
        # 第一部分为均值回归过程, 第二部分为布朗运动随机项
        dx = self.theta * (self.mu - x) * self.dt + self.std * np.sqrt(self.dt) * np.random.normal(size=self.action_shape)
        self.state = x + dx
        return np.clip(self.state * self.scale, -1 * self.bound, self.bound)

    def bound_decay(self):
        self.bound = max(self.bound * self.decay, 0.01)


if __name__ == "__main__":
    noise = OU_Noise(1, action_shape=[1])
    noise_list = []
    for _ in range(10000):
        noise_list.append(noise.get_noise())

    import matplotlib.pyplot as plt
    plt.plot(noise_list)
    plt.show()