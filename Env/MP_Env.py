import os
import gym
import pickle
import cloudpickle
import numpy as np
from multiprocessing import Pool, Pipe


class MP_Env():
    def __init__(self, env_name, worker_num=4):
        self.env_name = env_name
        self.worker_num = worker_num
        self.pool = Pool(self.worker_num)
        self.waiting = False
        self.closed = False
        self.env_list = []
        self.worker_list = []
        self.main_pipe_list = []
        self.worker_pipe_list = []
        for each in range(self.worker_num):
            env = gym.make(self.env_name).unwrapped
            wrapper_env = Wrapper_Env(env)
            main_pipe, worker_pipe = Pipe()
            worker = Worker(each, wrapper_env, main_pipe, worker_pipe)
            self.pool.apply_async(worker.run, args=())
            self.env_list.append(env)
            self.worker_list.append(worker)
            self.main_pipe_list.append(main_pipe)
            self.worker_pipe_list.append(worker_pipe)
        self.pool.close()

    def reset(self):
        state_list = []
        for main_pipe in self.main_pipe_list:
            main_pipe.send(("reset", None))
        for main_pipe in self.main_pipe_list:
            state_list.append(main_pipe.recv())
        return np.stack(state_list)

    def step(self, action_list):
        result_list = []
        for main_pipe, action in zip(self.main_pipe_list, action_list):
            main_pipe.send(("step", action))
        self.waiting = True
        for main_pipe in self.main_pipe_list:
            result_list.append(main_pipe.recv())
        self.waiting = False
        next_state_list, reward_list, done_list, _, _ = zip(*result_list)
        return np.stack(next_state_list), np.stack(reward_list), np.stack(done_list), _, _

    def close(self):
        if not self.closed:
            if self.waiting:
                for main_pipe in self.main_pipe_list:
                    main_pipe.recv()
            for main_pipe in self.main_pipe_list:
                main_pipe.send(("close", None))
            self.pool.terminate()
        else:
            pass


class Worker():
    def __init__(self, index, wrapper_env, main_pipe, worker_pipe):
        self.index = index
        self.env = wrapper_env.env
        self.main_pipe = main_pipe
        self.worker_pipe = worker_pipe
        self.step = 0

    def run(self):
        print("Worker:", self.index, "Process:", os.getpid())
        while True:
            cmd, data = self.worker_pipe.recv()
            if cmd == "reset":
                state, _ = self.env.reset()
                self.step = 0
                self.worker_pipe.send(state)
            elif cmd == "step":
                next_state, reward, done, _, _ = self.env.step(data)
                self.step += 1
                self.worker_pipe.send((next_state, reward, done, _ ,_))
            elif cmd == "close":
                self.env.close()
                self.worker_pipe.close()
                break
            else:
                raise NotImplementedError


class Wrapper_Env():
    def __init__(self, env):
        self.env = env

    def __getstate__(self):
        return cloudpickle.dumps(self.env)

    def __setstate__(self, env):
        self.env = pickle.loads(env)


if __name__ == "__main__":
    worker_num = 16
    mp_env = MP_Env("BipedalWalker-v3", worker_num=worker_num)
    for _ in range(1000):
        state_list = mp_env.reset()
        for _ in range(200):
            action_list = np.random.uniform(size=(worker_num, 4))
            next_state_list, reward_list, done_list, _, _ = mp_env.step(action_list)
            done_worker_list = np.where(done_list==True)[0]
            state_list = next_state_list
            for each in done_worker_list:
                print("worker", each, "reset")
                mp_env.main_pipe_list[each].send(("reset", None))
            for each in done_worker_list:
                state_list[each] = mp_env.main_pipe_list[each].recv()
    mp_env.close()
