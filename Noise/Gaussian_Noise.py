import numpy as np


class Gaussian_Noise():
    def __init__(self, index, action_shape, mu=0, std=0.4, scale=1, bound=1, decay=0.999):
        self.class_name = "Gaussian"
        self.index = index
        self.action_shape = action_shape
        # 噪声均值
        self.mu = mu
        # 噪声方差
        self.std = std
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
        self.state = np.random.normal(self.mu, self.std, size=self.action_shape)
        return np.clip(self.state * self.scale, -1 * self.bound, self.bound)

    def bound_decay(self):
        self.bound = max(self.bound * self.decay, 0.01)


if __name__ == "__main__":
    noise = Gaussian_Noise(1, action_shape=[1])
    noise_list = []
    for _ in range(10000):
        noise_list.append(noise.get_noise())

    import matplotlib.pyplot as plt
    plt.plot(noise_list)
    plt.show()