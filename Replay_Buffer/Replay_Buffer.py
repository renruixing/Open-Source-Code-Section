import pickle
import random
import numpy as np
from collections import deque
from Replay_Buffer.Segment_Tree import Sum_Tree


# 经验回放池
class Replay_Buffer():
    def __init__(self, buffer_size=1e5):
        self.buffer_size = int(buffer_size)
        self.buffer = deque(maxlen=self.buffer_size)

    # 保存一条经验
    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        self.buffer.append([state, action, log_prob, next_state, reward, done, dead])

    # 取出一批经验
    def sample(self, batch_size):
        memory_batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, done_batch, dead_batch = map(np.asarray, zip(*memory_batch))
        return state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, done_batch, dead_batch

    # 返回经验回放池大小
    def size(self):
        return len(self.buffer)

    # 重置经验回放池
    def reset(self):
        self.buffer.clear()

    # 保存经验回放池
    def save(self, agent_index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(agent_index), "wb") as f:
            pickle.dump(list(self.buffer), f)

    # 读取经验回放池
    def load(self, agent_index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(agent_index), "rb") as f:
            buffer = pickle.load(f)
        self.buffer.extend(buffer)


# 优先经验回放池
class Prioritized_Replay_Buffer():
    def __init__(self, buffer_size=1e5, alpha=0.6, beta=0.4, beta_increase=1e-3, min_priority=0.01, max_priority=1):
        self.buffer_size = int(buffer_size)
        self.sum_tree = Sum_Tree(self.buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        # 设置权重下限与上限
        self.min_priority = min_priority
        self.max_priority = max_priority

    # 保存一条经验
    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])
        if max_priority == 0:
            max_priority = self.max_priority
        self.sum_tree.add(max_priority, [state, action, log_prob, next_state, reward, done, dead])

    # 取出一批经验
    def sample(self, batch_size):
        index_batch = np.zeros((batch_size,), dtype=np.int32)
        state_batch = np.zeros((batch_size, self.sum_tree.data[0][0].size))
        action_batch = np.zeros((batch_size, self.sum_tree.data[0][1].size))
        log_prob_batch = np.zeros((batch_size,), dtype=np.float32)
        next_state_batch = np.zeros((batch_size, self.sum_tree.data[0][2].size))
        reward_batch = np.zeros((batch_size, ))
        done_batch = np.zeros((batch_size, ), dtype=np.bool)
        dead_batch = np.zeros((batch_size, ), dtype=np.bool)
        weight_batch = np.zeros((batch_size,))
        priority_segment = self.sum_tree.total_p() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increase])
        if self.sum_tree.full_tree:
            min_prob = np.min(self.sum_tree.tree[-self.sum_tree.capacity:]) / self.sum_tree.total_p()
        else:
            min_prob = np.min(self.sum_tree.tree[-self.sum_tree.capacity:self.sum_tree.capacity + self.sum_tree.data_pointer - 1]) / self.sum_tree.total_p()
        for each in range(batch_size):
            a, b = priority_segment * each, priority_segment * (each + 1)
            v = np.random.uniform(a, b)
            index, priority, memory = self.sum_tree.get_leaf(v)
            prob = priority / self.sum_tree.total_p()
            weight_batch[each] = np.power(prob / min_prob, -self.beta)
            index_batch[each] = index
            state_batch[each] = memory[0]
            action_batch[each] = memory[1]
            log_prob_batch[each] = memory[2]
            next_state_batch[each] = memory[3]
            reward_batch[each] = memory[4]
            done_batch[each] = memory[5]
            dead_batch[each] = memory[6]
        return state_batch, action_batch, log_prob_batch, next_state_batch, reward_batch, done_batch, dead_batch, index_batch, weight_batch

    # 更新经验权重
    def batch_update(self, tree_index_batch, TD_error_batch):
        TD_error_batch += self.min_priority
        TD_error_batch = np.minimum(TD_error_batch, self.max_priority)
        priority_batch = np.power(TD_error_batch, self.alpha)
        for tree_index, priority in zip(tree_index_batch, priority_batch):
            self.sum_tree.update(tree_index, priority)

    # 返回经验回放池大小
    def size(self):
        if self.sum_tree.full_tree:
            return self.sum_tree.capacity
        else:
            return self.sum_tree.data_pointer

    # 重置经验回放池
    def reset(self):
        self.sum_tree = Sum_Tree(self.buffer_size)

    # 保存经验回放池
    def save(self, agent_index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_weight.pickle".format(agent_index), "wb") as f:
            pickle.dump(self.sum_tree.tree, f)
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(agent_index), "wb") as f:
            pickle.dump(self.sum_tree.data, f)

    # 读取经验回放池
    def load(self, agent_index, file_path):
        with open(file_path + "/Agent_{}_Replay_Buffer_data.pickle".format(agent_index), "rb") as f:
            data = pickle.load(f)
        if len(data) != self.buffer_size:
            print("智能体" + str(agent_index) + "经验池不匹配")
        else:
            with open(file_path + "/Agent_{}_Replay_Buffer_weight.pickle".format(agent_index), "rb") as f:
                tree = pickle.load(f)
            self.sum_tree.tree = tree
            self.sum_tree.data = data
            self.sum_tree.full_tree = True