import numpy as np
import math as m


def transmission_rate(num, Tx_power_dBm, B_ratio, B_resource):
    """
    :param num:
    :param Tx_power_dBm: dBm
    :param B_ratio:
    :param B_resource: MHz
    noise: -174 dBm/Hz
    :return:
    """
    pass
    return sum_rate


class Task:
    def __init__(self, task_id, arrival_time, delay_threshold, vehicle_sensing_data, RSU_computation_data, UAV_to_RSU_data,
                 priority_weight):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.delay_threshold = delay_threshold
        self.vehicle_to_UAV_data = vehicle_sensing_data
        self.RSU_computation_data = RSU_computation_data
        self.UAV_to_RSU_data = UAV_to_RSU_data
        self.priority_weight = priority_weight

        self.rest_available_time = self.delay_threshold

        self.phase_1 = 0
        self.phase_3 = 0
        self.phase_2 = 0
        self.waiting_phase = 0


def Task_generation(num_task, num_timeslots):
    task = {}
    arrival_time = np.random.randint(0, num_timeslots * 0.5, num_task)
    delay_threshold = np.random.randint(300, 500, num_task)
    vehicle_sensing_data = np.random.randint(10, 20, num_task)  # 50MB ~ 150MB
    RSU_computation_data = np.random.randint(100, 200, num_task)  # MB
    UAV_to_RSU_data = np.random.randint(30, 40, num_task)  # MB
    priority_weight = np.zeros(1)  # init
    for i in range(0, num_task):
        task[i] = Task(task_id=i + 1, arrival_time=arrival_time[i], delay_threshold=delay_threshold[i],
                       vehicle_sensing_data=vehicle_sensing_data[i], RSU_computation_data=RSU_computation_data[i],
                       UAV_to_RSU_data=UAV_to_RSU_data[i], priority_weight=priority_weight)
    return task


class env:
    def __init__(self, num_tasks, num_timeslots):
        self.K = num_tasks  # number of vehicles N
        self.tao = 10  # length of time-slot tao: 10 ms
        self.T = num_timeslots  # number of time-slots
        self.R_UAV = 5  # uplink bandwidth of UAV: 5MHz
        self.R_RSU = 10  # downlink bandwidth of RSU: 10MHz
        self.R_c = 10  # computing resources of MEC server: 2.5 × 4 Core GHz
        self.P_k = 30  # Tx power of vehicles: 30 dBm
        self.P_UAV = 27  # Tx power of RSU: 27 dBm
        self.L = 5  # Q_exe length L: 5
        self.Task_done = [0] * self.K
        self.done = False
        self.num_vehicles = 10
        self.vehicle_positions = np.array([[np.random.randint(0, 100), np.random.randint(0, 100), 0]
                                          for _ in range(self.num_vehicles)])
        self.vehicle_velocities = np.array([[np.random.rand() * 10, np.random.rand() * 10, 0]
                                            for _ in range(self.num_vehicles)])
        self.vehicle_accelerations = np.array([[0, 0, 0] for _ in range(self.num_vehicles)])
        self.uav_position = np.array([50, 30, 20])
        self.rsu_position = np.array([50, 50, 0])

        # init_task
        self.task = Task_generation(num_task=self.K, num_timeslots=self.T)

        # init Q_exe state space
        self.time_state = [0] * self.L
        self.phase_1_state = [0] * self.L
        self.phase_3_state = [0] * self.L
        self.phase_2_state = [0] * self.L

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.L

        # Normalized norm
        self.T_value = []
        self.T_norm = 0
        self.D_1_value = []
        self.D_1_norm = 0
        self.D_3_value = []
        self.D_3_norm = 0
        self.D_2_value = []
        self.D_2_norm = 0
        for i in range(self.K):
            self.T_value.append(self.task[i].delay_threshold)
            self.D_1_value.append(self.task[i].vehicle_to_UAV_data)
            self.D_3_value.append(self.task[i].RSU_computation_data)
            self.D_2_value.append(self.task[i].UAV_to_RSU_data)
        self.T_norm = max(self.T_value)
        self.D_1_norm = max(self.D_1_value)
        self.D_3_norm = max(self.D_3_value)
        self.D_2_norm = max(self.D_2_value)

        self.task_service_delay = []

    def _update_vehicle_state(self):
        self.vehicle_positions += self.vehicle_velocities * self.tao + \
                                    0.5 * self.vehicle_accelerations * self.tao ** 2
        self.vehicle_velocities += self.vehicle_accelerations * self.tao

    def reset_state(self):
        self.vehicle_positions = np.array([[np.random.randint(0, 100), np.random.randint(0, 100), 0]
                                          for _ in range(self.num_vehicles)])
        self.vehicle_velocities = np.array([[np.random.rand() * 10, np.random.rand() * 10, 0]
                                            for _ in range(self.num_vehicles)])
        self.vehicle_accelerations = np.array([[0, 0, 0] for _ in range(self.num_vehicles)])
        self.uav_position = np.array([50, 30, 20])
        self.rsu_position = np.array([50, 50, 0])
        self.task = Task_generation(num_task=self.K, num_timeslots=self.T)
        # init Q_exe state space
        self.time_state = [0] * self.L
        self.phase_1_state = [0] * self.L
        self.phase_3_state = [0] * self.L
        self.phase_2_state = [0] * self.L

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.L

        self.Task_done = [0] * self.K
        self.done = False  # 所有任务完成则为1
        self.task_service_delay = []
        state = self.phase_1_state + self.phase_2_state + self.phase_3_state + self.time_state
        state = np.array(state)

    def get_task_queue(self, time_slot):
        """
        Define waiting queues Q_wait and execution queues Q_exe
        Q_wait length: infinite
        Q_exe length: self.L
        :return:
        """
        for k in range(self.K):
            # if t = t_k then add req_k into Q_wait
            if time_slot == self.task[k].arrival_time:
                self.Q_wait.append(k + 1)  # 1  3  5 ...
                self.task[k].waiting_phase = 1  # start waiting
                # print('The current time slot has arrived at task {}, which has been added to the task waiting queue'.format(i + 1))
        # print('Current task waiting queue:{}'.format(self.Q_wait))

        # init Q_wait_priority
        self.Q_wait_priority = []
        # init Q_wait queue state and get Q_wait queue task priority
        for j in range(len(self.Q_wait)):
            s_1 = self.task[self.Q_wait[j] - 1].vehicle_to_UAV_data
            s_3 = self.task[self.Q_wait[j] - 1].RSU_computation_data
            s_2 = self.task[self.Q_wait[j] - 1].UAV_to_RSU_data
            # If it is not equal, it indicates that it is already in the waiting queue
            if self.task[self.Q_wait[j] - 1].rest_available_time != self.task[self.Q_wait[j] - 1].delay_threshold:
                s_T = self.task[self.Q_wait[j] - 1].rest_available_time
            else:
                # Otherwise, the task has just entered the waiting queue
                s_T = self.task[self.Q_wait[j] - 1].delay_threshold

            # calculate the priority weight
            self.Q_wait_priority.append((s_1 + s_3 + s_2) / 3 / s_T)

        while True:
            is_available_Q_exe = 0
            if 0 in self.Q_exe:
                is_available_Q_exe = 1
            # Two conditions for offloading:
            # 1. There are tasks in the task waiting queue.
            # 2. There are available positions in the edge server running queue at RSU
            if is_available_Q_exe == 1 and len(self.Q_wait) != 0:
                # Obtain the system id of the task to be offloaded based on priority
                offloading_task_id = self.Q_wait[self.Q_wait_priority.index(max(self.Q_wait_priority))]
                free_index = self.Q_exe.index(0)
                self.Q_exe[free_index] = offloading_task_id

                self.task[offloading_task_id - 1].waiting_phase = 0
                self.task[offloading_task_id - 1].phase_1 = 1

                self.time_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].rest_available_time
                self.phase_1_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].vehicle_to_UAV_data
                self.phase_3_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].RSU_computation_data
                self.phase_2_state[self.Q_exe.index(offloading_task_id)] = self.task[offloading_task_id - 1].UAV_to_RSU_data

                self.Q_wait.remove(offloading_task_id)
                self.Q_wait_priority.remove(self.Q_wait_priority[self.Q_wait_priority.index(max(self.Q_wait_priority))])
            else:
                break

        for i in range(len(self.Q_wait)):
            self.task[self.Q_wait[i] - 1].rest_available_time -= self.tao
            # if self.task[self.Q_wait[i] - 1].rest_available_time <= 0:
            #     self.Q_wait.remove(self.Q_wait[i])
            #     self.Q_wait_priority.remove(self.Q_wait_priority[i])

        Q_exe_state = self.phase_1_state + self.phase_2_state + self.phase_3_state + self.time_state
        return Q_exe_state

    def step(self, action, step):
        self.P_k = 30  # Tx power of vehicles: 30 dBm
        self.P_UAV = 27  # Tx power of RSU: 27 dBm

        vehicle_to_UAV_B_ratio = action[:5]
        computing_resource_ratio = action[-5:]
        UAV_to_RSU_B_ratio = action[5:10]
        self._update_vehicle_state()
        vehicle_to_UAV_rate = transmission_rate(len(action[:5]), self.P_k, vehicle_to_UAV_B_ratio, self.R_UAV)
        computing_rate = []
        for i in range(len(computing_resource_ratio)):
            computing_rate.append(computing_resource_ratio[i] * self.R_c * 1e3 / 1e3)
        UAV_to_RSU_rate = transmission_rate(len(action[5:10]), self.P_UAV, UAV_to_RSU_B_ratio, self.R_RSU)

        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] != 0:
                self.task[self.Q_exe[i] - 1].rest_available_time -= self.tao



                self.time_state[i] = self.task[self.Q_exe[i] - 1].rest_available_time
                # vehicle_to_UAV_rate: Mbits/ms
                self.phase_1_state[i] -= self.task[self.Q_exe[i] - 1].phase_1 * self.tao * vehicle_to_UAV_rate[i]
                # computing_rate: Mcycles/ms
                self.phase_3_state[i] -= self.task[self.Q_exe[i] - 1].phase_3 * self.tao * computing_rate[i]
                self.phase_2_state[i] -= self.task[self.Q_exe[i] - 1].phase_2 * self.tao * UAV_to_RSU_rate[i]

                if self.time_state[i] <= 0 and self.phase_3_state[i] > 0:
                    self.Task_done[self.Q_exe[i] - 1] = -1
                    self.task_service_delay.append(self.task[self.Q_exe[i] - 1].delay_threshold)
                    self.Q_exe[i] = 0
                    self.time_state[i] = 0
                    self.phase_1_state[i] = 0
                    self.phase_3_state[i] = 0
                    self.phase_2_state[i] = 0
                else:
                    if self.phase_1_state[i] <= 0:
                        self.phase_1_state[i] = 0
                        self.task[self.Q_exe[i] - 1].phase_1 = 0
                        self.task[self.Q_exe[i] - 1].phase_2 = 1

                    if self.phase_2_state[i] <= 0:
                        self.phase_2_state[i] = 0
                        self.task[self.Q_exe[i] - 1].phase_2 = 0
                        self.task[self.Q_exe[i] - 1].phase_3 = 1

                    if self.phase_3_state[i] <= 0:
                        self.task_service_delay.append(self.task[self.Q_exe[i] - 1].delay_threshold - self.task[self.Q_exe[i] - 1].rest_available_time)
                        self.phase_3_state[i] = 0
                        self.time_state[i] = 0
                        self.task[self.Q_exe[i] - 1].phase_3 = 0
                        self.Task_done[self.Q_exe[i] - 1] = 1
                        self.Q_exe[i] = 0

        new_state = self.phase_1_state + self.phase_2_state + self.phase_3_state + self.time_state
        while 0 not in self.Task_done:
            self.done = True
            break

        return new_state, self.done


