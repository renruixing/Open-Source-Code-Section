import numpy as np


# 线性树结构
class Sum_Tree(object):
    def __init__(self, capacity):
        # 叶节点数量
        self.capacity = capacity
        # 线性树节点数量, 枝节点self.capacity - 1个, 叶节点self.capacity个
        self.tree = np.zeros(shape=(2 * self.capacity - 1,))
        self.data = np.zeros(shape=(self.capacity), dtype=object)
        self.data_pointer = 0
        self.full_tree = False

    # 添加数据, 传入采样概率
    def add(self, p, data):
        # 挂载叶节点
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # 更新树权重
        self.update(tree_idx, p)
        self.data_pointer += 1
        # 保证数据先进先出
        if self.data_pointer >= self.capacity:
            self.full_tree = True
            self.data_pointer = 0

    # 更新权重
    def update(self, tree_idx, p):
        # 获取权重改变量
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # 获取父节点, 并更新权重
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 获取叶节点
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            # 获取左右子节点
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # 若左节点为最后一个（没有右节点则为最后一层）
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                # 左节点权重直接用
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                # 右节点需要减去左节点权重
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        # 获取数据索引
        data_idx = leaf_idx - self.capacity + 1
        # 返回节点编号、节点权重与数据
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    # 返回线性树总权重
    def total_p(self):
        return self.tree[0]