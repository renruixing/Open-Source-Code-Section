# average service time delay comparison: proposed scheme VS DDPG scheme VS average allocation VS random allocation
import matplotlib.pyplot as plt
import numpy as np

proposed_scheme = [0.982, 0.862, 0.820, 0.784, 0.741, 0.694, 0.624]
single_critic_scheme = [0.848, 0.755, 0.692, 0.645, 0.594, 0.556, 0.485]
average_allocation_scheme = [0.502, 0.4267, 0.3512, 0.32, 0.285, 0.267, 0.2523]
random_allocation_scheme = [0.355, 0.295, 0.2378, 0.2084, 0.1717, 0.1485, 0.1414]

x = np.arange(10, 24, 2)
fig, ax = plt.subplots()

ax.plot(x, proposed_scheme, 'yd-', linewidth=1, label='Proposed scheme')
ax.plot(x, single_critic_scheme, 'g*-', linewidth=1, label='Single Critic scheme')
ax.plot(x, average_allocation_scheme, 'bx-', linewidth=1, label='Average-allocation scheme')
ax.plot(x, random_allocation_scheme, 'ms-', linewidth=1, label='Random-allocation scheme')

plt.grid()
plt.legend(loc='upper right', prop={'size': 10})  # 显示标签
plt.xlabel('Number of vehicles', fontsize=12)  # 设置X轴名称
plt.ylabel('Average task completion rate', fontsize=12)  # 设置Y轴名称
plt.savefig('Testing_Average_completion_rate.pdf', dpi=1200, bbox_inches='tight')
plt.show()

