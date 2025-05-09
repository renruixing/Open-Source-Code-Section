import time
import numpy as np
import tensorflow as tf
from Agent.PG.PG import PG_Agent
from Agent.AC.AC import AC_Agent
from Agent.A2C.A2C import A2C_Agent
from Agent.PPO.PPO_Clip import PPO_Agent
from Agent.DQN.DQN import DQN_Agent
from Agent.DDQN.DDQN import DDQN_Agent
from Agent.Duel_DQN.Duel_DQN import Duel_DQN_Agent
from Agent.D3QN.D3QN import D3QN_Agent
from Agent.DDPG.DDPG import DDPG_Agent
from Agent.TD3.TD3 import TD3_Agent
from Agent.SAC.SAC_V2 import SAC_Agent
from Utils.Common import set_seed
from Env.test_Env import env
import random


# 设置TF2数据类型
tf.keras.backend.set_floatx("float32")


class Test():
    def __init__(self):
        # region 实验参数
        # 设置智能体编号
        self.agent_index = 1
        # 设置智能体类型
        self.agent_class = "DDPG"
        # 设置优先经验回放
        self.prioritized_replay = False  # 1.如果算法足够好，没必要用PER；2.如果奖励函数足够dense，也没必要用PER；3.用了PER速度大大降低，性能不一定会提高
        # 设置环境
        self.env_name = "train-Env"

        # 设置实验随机种子
        self.exp_seed = 316
        set_seed(self.exp_seed)
        # 设置实验时间
        # self.exp_time = "/" + time.strftime("%Y-%m-%d %H-%M-%S")

        # Proposed: 2024-12-07 17-42-15   lr1e-3_batchsize128
        self.exp_name = self.agent_class + "/" + self.env_name + "/2024-12-07 17-42-15"

        # 设置回合次数
        self.episode_num = 100
        # 设置单回合最大步数
        self.step_num = 100
        # endregion

        # region 路径参数
        # 设置日志存储路径
        # self.log_save_path = "Logs/" + self.exp_name  # / 卡了我一个小时 已解决
        self.log_save_path = None
        # 设置模型存储路径
        # self.model_save_path = "Models/" + self.exp_name
        self.model_save_path = None
        # 设置经验池存储路径
        # self.buffer_save_path = "Models/" + self.exp_name
        self.buffer_save_path = None
        # 设置模型加载路径
        self.model_load_path = "Models/" + self.exp_name
        # self.model_load_path = None
        # 设置经验池加载路径
        # self.buffer_load_path = "Models/" + self.exp_name
        self.buffer_load_path = None
        # endregion

        # region 经验回放池参数
        # 设置训练起始batch数
        self.start_batch = 10
        # 设置batch大小
        self.batch_size = 128
        # 设置buffer大小
        self.buffer_size = 1e5
        # endregion

        # region 模型参数
        # 设置Actor模型结构
        self.actor_unit_num_list = [128, 64]
        # 设置Actor模型激活函数
        self.actor_activation = "softmax"
        # 设置Actor模型学习率(Adam)
        self.actor_lr = 1e-3

        # 设置Critic模型结构
        self.critic_unit_num_list = [128, 64, 64]
        # 设置Critic模型激活函数
        self.critic_activation = "linear"
        # 设置Critic模型学习率(默认: Adam)
        self.critic_lr = 1e-3

        # 设置Critic网络训练频率(单位: sum_step):
        self.train_freq = 1
        # 设置Actor网络训练频率(单位: train_freq)
        self.actor_train_freq = 2
        # 设置模型存储频率(单位: episode)
        self.save_freq = 100
        # 设置模型更新频率(单位: train_freq)
        self.update_freq = 100
        # 设置模型更新权重
        self.tau = 0.05
        # endregion

        # region 奖励参数
        # 奖励奖励衰减系数
        self.reward_gamma = 0.98
        # 设置奖励放大系数
        self.reward_scale = 1
        # endregion

        # region 探索策略参数
        # 设置Target探索
        self.target_action = True

        # 设置随机探索权重
        self.epsilon = 0.9
        self.min_epsilon = 1e-2
        # 设置随机探索衰减系数
        self.epsilon_decay = 1e-4
        # endregion

        # region PPO智能体参数
        # 设置优势函数系数
        self.lamba = 0.95
        # 设置单次训练次数(单位: train_freq)
        self.train_epoch = 10
        # 设置概率裁切边界
        self.clip_epsilon = 0.2
        # endregion

        # region TD3智能体参数
        # 设置评估噪音方差(默认: Gaussian, 均值为0)
        self.eval_noise_std = 0.2
        # 设置评估噪音放大系数
        self.eval_noise_scale = 1
        # 设置评估噪音边界
        self.eval_noise_bound = 0.2
        # endregion

        # region SAC智能体参数
        # 设置自适应熵
        self.adaptive_entropy_alpha = True
        # 设置初始熵系数
        self.entropy_alpha = 0.2
        # 设置熵系数学习率
        self.entropy_alpha_lr = 3e-4
        # endregion

        # 任务随机到达率---ren set 03.9.8
        # self.gamma = 0.8
        # 创建环境
        print("创建环境")
        self.number_task = 10
        self.env = env(num_tasks=self.number_task, num_timeslots=self.step_num)

        # 设置状态空间维度
        # [vehicle => UAV state, UAV => RSU state, computation state, time state]
        self.state_shape = [5, 5, 5, 5]
        # 设置动作空间维度
        # [vehicle => UAV bandwidth ratio, UAV => RSU bandwidth ratio, computing resource ratio]
        self.action_shape = [5, 5, 5]

        # 创建智能体
        print("创建智能体")
        self.agent = self.agent_create()

        # 加载模型
        if self.model_load_path != None:
            print("加载模型")
            self.agent.model_load(self.model_load_path)

    # 获取动作
    def get_action(self, state):
        # 智能体探索策略
        if self.target_action:
            action, log_prob = self.agent.get_target_action(state)
        else:
            action, log_prob = self.agent.get_action(state)
        return action, log_prob

    # 添加探索噪音
    def add_noise(self, action):
        """
        :param action: shape: [5, 5, 5] => 1 * 15
        :return:
        """
        noise = self.explore_noise.get_noise()
        noise_action = noise + action
        return np.clip(noise_action, -1, 1)

    # 更新随机探索权重
    def update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon - self.epsilon_decay
        else:
            self.epsilon = self.min_epsilon
        return self.epsilon

    def generate_random_numbers(self):
        numbers = [random.uniform(0, 1) for _ in range(4)]
        fifth_number = 1 - sum(numbers)
        if 0 <= fifth_number <= 1:
            numbers.append(fifth_number)
            return numbers
        else:
            return self.generate_random_numbers()

    # 训练智能体
    def test(self):
        task_completion_rate_list = []
        task_delay_list = []
        for episode in range(self.episode_num):
            sum_reward = 0
            step = 0
            self.env.reset_state()
            done = False
            while not done:
                Q_exe_state = self.env.get_task_queue(step)
                # proposed scheme
                action, log_prob = self.get_action(Q_exe_state)

                # single_critic scheme
                # action, log_prob = self.get_action(Q_exe_state)

                # average allocation scheme
                # action = [0.2] * 15

                # random allocation scheme
                # action_1 = self.generate_random_numbers()
                # action_2 = self.generate_random_numbers()
                # action_3 = self.generate_random_numbers()
                # action = action_1 + action_2 + action_3

                act = action
                step += 1
                next_state, done = self.env.step(act, step)
                if step >= self.step_num:
                    done = True

            task_done = 0
            for j in range(len(self.env.Task_done)):
                if self.env.Task_done[j] == 1:
                    task_done += 1
            task_completion_rate = task_done / self.number_task
            task_completion_rate_list.append(task_completion_rate)
            task_delay_list.append(np.mean(self.env.task_service_delay))
        print("number_tasks:", self.number_task, "average_task_completion_rate:", np.mean(task_completion_rate_list),
              "average_task_latency:", np.mean(task_delay_list))

    def agent_create(self):
        agent = None
        if self.agent_class == "PG":
            agent = PG_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                             actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                             gamma=self.reward_gamma, buffer_size=self.buffer_size)
        elif self.agent_class == "AC":
            agent = AC_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                             actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                             critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                             gamma=self.reward_gamma, buffer_size=self.buffer_size)
        elif self.agent_class == "A2C":
            agent = A2C_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                              actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                              critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                              gamma=self.reward_gamma, buffer_size=self.buffer_size, lamba=self.lamba)
        elif self.agent_class == "PPO":
            agent = PPO_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                              actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                              critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                              gamma=self.reward_gamma, tau=self.tau, buffer_size=self.buffer_size,
                              lamba=self.lamba, train_epoch=self.train_epoch, clip_epsilon=self.clip_epsilon)
        elif self.agent_class == "DQN":
            agent = DQN_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                              critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                              update_freq=self.update_freq, gamma=self.reward_gamma, tau=self.tau,
                              batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False)
        elif self.agent_class == "DDQN":
            agent = DDQN_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                               critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                               update_freq=self.update_freq, gamma=self.reward_gamma, tau=self.tau,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False)
        elif self.agent_class == "Duel_DQN":
            agent = Duel_DQN_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                                   critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                                   update_freq=self.update_freq, gamma=self.reward_gamma, tau=self.tau,
                                   batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False)
        elif self.agent_class == "D3QN":
            agent = D3QN_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                               critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                               update_freq=self.update_freq, gamma=self.reward_gamma, tau=self.tau,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False)
        elif self.agent_class == "DDPG":
            agent = DDPG_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                               actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                               critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                               update_freq=self.update_freq, actor_train_freq=self.actor_train_freq, gamma=self.reward_gamma, tau=self.tau,
                               batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False)
        elif self.agent_class == "TD3":
            agent = TD3_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                              actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                              critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                              update_freq=self.update_freq, actor_train_freq=self.actor_train_freq, gamma=self.reward_gamma, tau=self.tau,
                              batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False,
                              eval_noise_std=self.eval_noise_std, eval_noise_scale=self.eval_noise_scale, eval_noise_bound=self.eval_noise_bound)
        elif self.agent_class == "SAC":
            agent = SAC_Agent(agent_index=self.agent_index, state_shape=self.state_shape, action_shape=self.action_shape,
                              actor_unit_num_list=self.actor_unit_num_list, actor_activation=self.actor_activation, actor_lr=self.actor_lr,
                              critic_unit_num_list=self.critic_unit_num_list, critic_activation=self.critic_activation, critic_lr=self.critic_lr,
                              update_freq=self.update_freq, actor_train_freq=self.actor_train_freq, gamma=self.reward_gamma, tau=self.tau,
                              batch_size=self.batch_size, buffer_size=self.buffer_size, prioritized_replay=False,
                              adaptive_entropy_alpha=self.adaptive_entropy_alpha, entropy_alpha=self.entropy_alpha, entropy_alpha_lr=self.entropy_alpha_lr)
        return agent


if __name__ == "__main__":
    train = Test()
    train.test()
