import numpy as np


def env_adapter(env_name, state, action, next_state, reward, done):
    dead = done
    if env_name == "CartPole-v1":
        pass
    elif env_name == "MountainCar-v0":
        if dead:
            reward += 10
        if state[0] >= 0.4:
            reward += 0.5
    elif env_name == "MountainCarContinuous-v0":
        pass
    elif env_name == "Pendulum-v1":
        reward = (reward + 8) / 8
    elif env_name == "LunarLanderContinuous-v2":
        pass
    elif env_name == "BipedalWalker-v3" or env_name == "BipedalWalkerHardcore-v3":
        if reward <= -100:
            reward = -1
    return state, action, next_state, reward, done, dead


def action_adapter(env_name, action):
    act = action
    if env_name == "CartPole-v1":
        act = np.argmax(act)
    elif env_name == "MountainCar-v0":
        act = np.argmax(act)
    elif env_name == "MountainCarContinuous-v0":
        pass
    elif env_name == "Pendulum-v1":
        act = action * 2
    elif env_name == "LunarLanderContinuous-v2":
        pass
    elif env_name == "BipedalWalker-v3" or env_name == "BipedalWalkerHardcore-v3":
        pass
    return act, action
