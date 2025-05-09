import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict


def _tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = smooth + 1
    x = values[0]
    res = [x]
    for i in range(1, len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


def different_lr_reward():
    fig, ax = plt.subplots()
    # lr=1e-2 batch size = 128
    A_scheme = pd.read_csv('./lr1e-2_batchsize128/run-.-tag-Train_Rewards.csv')
    A_scheme_rewards = A_scheme[['Value']]
    A_scheme_rewards_list = _tensorboard_smoothing(np.round(A_scheme_rewards.values.tolist(), 2), 0.5)
    A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.95)

    # lr=1e-3 batch size = 128
    B_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Train_Rewards.csv')
    B_scheme_rewards = B_scheme[['Value']]
    B_scheme_rewards_list = _tensorboard_smoothing(np.round(B_scheme_rewards.values.tolist(), 2), 0.5)
    B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.95)

    # lr=1e-4 batch size = 128
    C_scheme = pd.read_csv('./lr1e-4_batchsize128/run-.-tag-Train_Rewards.csv')
    C_scheme_rewards = C_scheme[['Value']]
    C_scheme_rewards_list = _tensorboard_smoothing(np.round(C_scheme_rewards.values.tolist(), 2), 0.5)
    C_scheme_smooth_rewards = _tensorboard_smoothing(np.round(C_scheme_rewards_list, 2), 0.95)

    x = np.arange(0, len(A_scheme_rewards), 1)

    ax.plot(x, A_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, A_scheme_smooth_rewards, label="learning rate = 1e-2", color="#FF7043", zorder=6)

    ax.plot(x, B_scheme_rewards_list, color="#FF00FF", alpha=0.2, zorder=2)
    ax.plot(x, B_scheme_smooth_rewards, label="learning rate = 1e-3", color="#FF00FF", zorder=1)

    ax.plot(x, C_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, C_scheme_smooth_rewards, label="learning rate = 1e-4", color="#0000FF", zorder=5)

    # ax.plot(x, C_scheme_rewards, color="#90EE90", alpha=0.3, zorder=1)
    # ax.plot(x, C_scheme_smooth_rewards, label="C scheme", color="#00FF00", zorder=4)

    # ax.plot(x, D_scheme_rewards, color="#E6BEFF", alpha=0.3, zorder=1)

    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 12})  # 显示标签
    plt.savefig('./different_lr_reward.pdf')
    plt.show()


def different_lr_completion_rate():
    fig, ax = plt.subplots()
    # lr=1e-2 batch size = 128
    A_scheme = pd.read_csv('./lr1e-2_batchsize128/run-.-tag-Train_Completion Rate.csv')
    A_scheme_rewards = A_scheme[['Value']]
    A_scheme_rewards_list = _tensorboard_smoothing(np.round(A_scheme_rewards.values.tolist(), 2), 0.5)
    A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.95)

    # lr=1e-3 batch size = 128
    B_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Train_Completion Rate.csv')
    B_scheme_rewards = B_scheme[['Value']]
    B_scheme_rewards_list = _tensorboard_smoothing(np.round(B_scheme_rewards.values.tolist(), 2), 0.5)
    B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.95)

    # lr=1e-4 batch size = 128
    C_scheme = pd.read_csv('./lr1e-4_batchsize128/run-.-tag-Train_Completion Rate.csv')
    C_scheme_rewards = C_scheme[['Value']]
    C_scheme_rewards_list = _tensorboard_smoothing(np.round(C_scheme_rewards.values.tolist(), 2), 0.5)
    C_scheme_smooth_rewards = _tensorboard_smoothing(np.round(C_scheme_rewards_list, 2), 0.95)

    x = np.arange(0, len(A_scheme_rewards), 1)

    ax.plot(x, A_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, A_scheme_smooth_rewards, label="learning rate = 1e-2", color="#FF7043", zorder=6)

    ax.plot(x, B_scheme_rewards_list, color="#FF00FF", alpha=0.2, zorder=2)
    ax.plot(x, B_scheme_smooth_rewards, label="learning rate = 1e-3", color="#FF00FF", zorder=1)

    ax.plot(x, C_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, C_scheme_smooth_rewards, label="learning rate = 1e-4", color="#0000FF", zorder=5)

    # ax.plot(x, C_scheme_rewards, color="#90EE90", alpha=0.3, zorder=1)
    # ax.plot(x, C_scheme_smooth_rewards, label="C scheme", color="#00FF00", zorder=4)
    # ax.plot(x, D_scheme_rewards, color="#E6BEFF", alpha=0.3, zorder=1)
    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Completion rates', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 12})  # 显示标签
    plt.savefig('./different_lr_completion_rate.pdf')
    plt.show()


def different_batch_size_reward():
    # lr = 1e-3
    fig, ax = plt.subplots()
    # batch size = 64
    A_scheme = pd.read_csv('./lr1e-3_batchsize64/run-.-tag-Train_Rewards.csv')
    A_scheme_rewards = A_scheme[['Value']]
    A_scheme_rewards_list = _tensorboard_smoothing(np.round(A_scheme_rewards.values.tolist(), 2), 0.5)
    A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.95)

    # batch size = 128
    B_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Train_Rewards.csv')
    B_scheme_rewards = B_scheme[['Value']]
    B_scheme_rewards_list = _tensorboard_smoothing(np.round(B_scheme_rewards.values.tolist(), 2), 0.5)
    B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.95)

    # batch size = 256
    C_scheme = pd.read_csv('./lr1e-3_batchsize256/run-.-tag-Train_Rewards.csv')
    C_scheme_rewards = C_scheme[['Value']]
    C_scheme_rewards_list = _tensorboard_smoothing(np.round(C_scheme_rewards.values.tolist(), 2), 0.5)
    C_scheme_smooth_rewards = _tensorboard_smoothing(np.round(C_scheme_rewards_list, 2), 0.95)

    x = np.arange(0, len(A_scheme_rewards), 1)

    ax.plot(x, A_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, A_scheme_smooth_rewards, label="batch size = 64", color="#FF7043", zorder=6)

    ax.plot(x, B_scheme_rewards_list, color="#FF00FF", alpha=0.2, zorder=2)
    ax.plot(x, B_scheme_smooth_rewards, label="batch size = 128", color="#FF00FF", zorder=1)

    ax.plot(x, C_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, C_scheme_smooth_rewards, label="batch size = 256", color="#0000FF", zorder=5)

    # ax.plot(x, C_scheme_rewards, color="#90EE90", alpha=0.3, zorder=1)
    # ax.plot(x, C_scheme_smooth_rewards, label="C scheme", color="#00FF00", zorder=4)
    # ax.plot(x, D_scheme_rewards, color="#E6BEFF", alpha=0.3, zorder=1)
    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    plt.grid()
    plt.legend(loc='lower right', prop={'size': 12})  # 显示标签
    plt.savefig('./different_batch_size_reward.pdf')
    plt.show()


def different_lr_Actor_loss():
    fig, ax = plt.subplots()
    # lr = 1e-2 batch size = 128
    A_scheme = pd.read_csv('./lr1e-2_batchsize128/run-.-tag-Loss_Actor.csv')
    A_scheme_rewards = A_scheme[['Value']]
    A_scheme_rewards_list = A_scheme_rewards.values.tolist()
    A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.8)

    # lr = 1e-3 batch size = 128
    B_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Loss_Actor.csv')
    B_scheme_rewards = B_scheme[['Value']]
    B_scheme_rewards_list = B_scheme_rewards.values.tolist()
    B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.8)

    # lr = 1e-4 batch size = 128
    C_scheme = pd.read_csv('./lr1e-4_batchsize128/run-.-tag-Loss_Actor.csv')
    C_scheme_rewards = C_scheme[['Value']]
    C_scheme_rewards_list = C_scheme_rewards.values.tolist()
    C_scheme_smooth_rewards = _tensorboard_smoothing(np.round(C_scheme_rewards_list, 2), 0.6)

    x = np.arange(0, len(A_scheme_rewards), 1)

    ax.plot(x, A_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, A_scheme_smooth_rewards, label="learning rate = 0.01", color="#FF7043", zorder=6)

    ax.plot(x, B_scheme_rewards_list, color="#FF00FF", alpha=0.2, zorder=2)
    ax.plot(x, B_scheme_smooth_rewards, label="learning rate = 0.001", color="#FF00FF", zorder=1)

    ax.plot(x, C_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, C_scheme_smooth_rewards, label="learning rate = 0.0001", color="#0000FF", zorder=5)

    # ax.plot(x, C_scheme_rewards, color="#90EE90", alpha=0.3, zorder=1)
    # ax.plot(x, C_scheme_smooth_rewards, label="C scheme", color="#00FF00", zorder=4)
    # ax.plot(x, D_scheme_rewards, color="#E6BEFF", alpha=0.3, zorder=1)
    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Actor loss', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 11})  # 显示标签
    plt.savefig('./different_lr_Actor_loss.pdf')
    plt.show()


def Proposed_scheme_Critic_1_and_2_loss():
    # lr = 1e-3 batch size = 128
    fig, ax = plt.subplots()
    A_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Loss_Critic 1.csv')
    A_scheme_rewards = A_scheme[['Value']]
    A_scheme_rewards_list = A_scheme_rewards.values.tolist()
    A_scheme_smooth_rewards = _tensorboard_smoothing(np.round(A_scheme_rewards_list, 2), 0.85)

    B_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Loss_Critic 2.csv')
    B_scheme_rewards = B_scheme[['Value']]
    B_scheme_rewards_list = B_scheme_rewards.values.tolist()
    B_scheme_smooth_rewards = _tensorboard_smoothing(np.round(B_scheme_rewards_list, 2), 0.85)

    x = np.arange(0, len(A_scheme_rewards), 1)

    ax.plot(x, A_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, A_scheme_smooth_rewards, label="Critic 1 loss", color="#FF7043", zorder=6)

    ax.plot(x, B_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, B_scheme_smooth_rewards, label="Critic 2 loss", color="#0000FF", zorder=1)

    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Critic 1 and Critic 2 loss', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 11})  # 显示标签
    plt.savefig('./Proposed_scheme_Critic_1_and_2_loss.pdf')
    plt.show()


def Different_scheme_reward():
    # lr = 1e-3 batch size = 128
    fig, ax = plt.subplots()

    Proposed_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Train_Rewards.csv')
    Proposed_scheme_rewards = Proposed_scheme[['Value']]
    Proposed_scheme_rewards_list = _tensorboard_smoothing(np.round(Proposed_scheme_rewards.values.tolist(), 2), 0.5)
    Proposed_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Proposed_scheme_rewards_list, 2), 0.95)

    DDPG_scheme = pd.read_csv('./DDPG_lr1e-3_batchsize128/run-.-tag-Train_Rewards.csv')
    DDPG_scheme_rewards = DDPG_scheme[['Value']]
    DDPG_scheme_rewards_list = _tensorboard_smoothing(np.round(DDPG_scheme_rewards.values.tolist(), 2), 0.5)
    DDPG_scheme_smooth_rewards = _tensorboard_smoothing(np.round(DDPG_scheme_rewards_list, 2), 0.85)

    x = np.arange(0, len(Proposed_scheme_rewards), 1)

    ax.plot(x, Proposed_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, Proposed_scheme_smooth_rewards, label="Proposed scheme", color="#FF7043", zorder=6)

    ax.plot(x, DDPG_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, DDPG_scheme_smooth_rewards, label="Single Critic scheme", color="#0000FF", zorder=1)

    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 11})  # 显示标签
    plt.savefig('./Different_scheme_reward.pdf')
    plt.show()


def Different_scheme_completion_rate():
    # lr = 1e-3 batch size = 128
    fig, ax = plt.subplots()

    Proposed_scheme = pd.read_csv('./lr1e-3_batchsize128/run-.-tag-Train_Completion Rate.csv')
    Proposed_scheme_rewards = Proposed_scheme[['Value']]
    Proposed_scheme_rewards_list = _tensorboard_smoothing(np.round(Proposed_scheme_rewards.values.tolist(), 2), 0.5)
    Proposed_scheme_smooth_rewards = _tensorboard_smoothing(np.round(Proposed_scheme_rewards_list, 2), 0.95)

    DDPG_scheme = pd.read_csv('./DDPG_lr1e-3_batchsize128/run-.-tag-Train_Completion Rate.csv')
    DDPG_scheme_rewards = DDPG_scheme[['Value']]
    DDPG_scheme_rewards_list = _tensorboard_smoothing(np.round(DDPG_scheme_rewards.values.tolist(), 2), 0.5)
    DDPG_scheme_smooth_rewards = _tensorboard_smoothing(np.round(DDPG_scheme_rewards_list, 2), 0.85)

    x = np.arange(0, len(Proposed_scheme_rewards), 1)

    ax.plot(x, Proposed_scheme_rewards_list, color="#FF7043", alpha=0.2, zorder=3)
    ax.plot(x, Proposed_scheme_smooth_rewards, label="Proposed scheme", color="#FF7043", zorder=6)

    ax.plot(x, DDPG_scheme_rewards_list, color="#0000FF", alpha=0.2, zorder=2)
    ax.plot(x, DDPG_scheme_smooth_rewards, label="Single Critic scheme", color="#0000FF", zorder=1)

    default_ticks = np.linspace(0, 1000, 5)
    new_ticks = np.linspace(0, 40000, 5)
    x_ticklabels = [str(int(tick)) for tick in new_ticks]
    plt.xticks(default_ticks, x_ticklabels)

    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Completion rates', fontsize=12)
    plt.grid()
    plt.legend(loc='best', prop={'size': 11})  # 显示标签
    plt.savefig('./Different_scheme_completion_rate.pdf')
    plt.show()


# 实验结果的第一张图
# different_batch_size_reward()

# 实验结果第二张图 - 奖励
# different_lr_reward()

# 实验结果第三张图 - 完成率
# different_lr_completion_rate()

# 实验结果的第四张图  不同学习率下的Actor Loss
# different_lr_Actor_loss()

# 实验结果的第五张图 Critic 1 & 2的Loss
# Proposed_scheme_Critic_1_and_2_loss()

# 实验结果的第六、七张图 - 不同方案下完成率和奖励的对比 - Single Critic 线性分配 所提方案
Different_scheme_reward()
Different_scheme_completion_rate()
