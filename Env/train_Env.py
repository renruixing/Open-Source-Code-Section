import random
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


class Task:
    def __init__(self, task_id, arrival_time, delay_threshold, vehicle_sensing_data, computation_data, download_data,
                 priority_weight):
        self.task_id = task_id
        self.arrival_time = arrival_time
        self.delay_threshold = delay_threshold
        self.upload_data = vehicle_sensing_data
        self.computation_data = computation_data
        self.download_data = download_data
        self.priority_weight = priority_weight

        self.rest_available_time = self.delay_threshold

        self.upload_phase = 0
        self.computing_phase = 0
        self.download_phase = 0
        self.waiting_phase = 0


def Task_generation(num_task, num_timeslots, compression_coefficient):
    task = {}
    arrival_time = 0
    delay_threshold = np.random.randint(50, 500, num_task)
    vehicle_sensing_data = np.random.randint(50, 100, num_task)  # 50MB ~ 150MB  vehicle => UAV
    computation_data = np.random.randint(100, 200, num_task)  # MB
    fusion_sensing_data = np.random.randint(100, 150, num_task)  # MB  UAV => RSU
    priority_weight = np.zeros(1)  # init
    for i in range(0, num_task):
        task[i] = Task(task_id=i + 1, arrival_time=arrival_time, delay_threshold=delay_threshold[i],
                       vehicle_sensing_data=vehicle_sensing_data[i], computation_data=computation_data[i],
                       download_data=fusion_sensing_data[i], priority_weight=priority_weight)
    return task


class env:
    def __init__(self, gamma):
        self.tao = 10  # length of time-slot tao: 10ms
        # self.T = num_timeslots  # number of time-slots
        self.R_UAV = 5  # uplink bandwidth of UAV: 5MHz
        self.R_RSU = 10  # downlink bandwidth of RSU: 10MHz
        self.R_c = 20  # computing resources of MEC server: 2.5 × 8 Core GHz
        self.P_k = 30  # Tx power of vehicles: 30 dBm
        self.P_UAV = 27  # Tx power of RSU: 27 dBm
        self.L = 5  # Q_exe length L: 5
        self.done = 0
        self.gamma = gamma

        # init_task
        # self.task = Task_generation(num_task=self.N, num_timeslots=self.T, compression_coefficient=self.r_sep)

        # init Q_exe state space
        self.time_state = [0] * self.L
        self.phase_1_state = [0] * self.L
        self.phase_3_state = [0] * self.L
        self.phase_2_state = [0] * self.L

        # init task queues
        self.Q_wait = []
        self.Q_wait_priority = []
        self.Q_exe = [0] * self.L

        self.num_tasks = 0
        self.Task_done = [0] * self.L

        # init phase
        self.phase_1 = [0] * self.L
        self.phase_3 = [0] * self.L
        self.phase_2 = [0] * self.L

        # init task size
        self.vehicle_to_UAV_data = [0] * self.L
        self.RSU_computation_data = [0] * self.L
        self.UAV_to_RSU_data = [0] * self.L
        self.delay_constraint = [0] * self.L

    def reset_state(self):
        self.L = 5  # Q_exe length M: 5
        self.Q_exe = [0] * self.L
        self.num_tasks = 0
        self.Task_done = [0] * self.L

        # init phase
        self.phase_1 = [0] * self.L  # c_{k,U}
        self.phase_3 = [0] * self.L  # c_{k,R}
        self.phase_2 = [0] * self.L  # c_{k,c}

        # init task size
        self.vehicle_to_UAV_data = [0] * self.L
        self.RSU_computation_data = [0] * self.L
        self.UAV_to_RSU_data = [0] * self.L
        self.delay_constraint = [0] * self.L

        # init Q_exe state space
        self.time_state = [0] * self.L
        self.phase_1_state = [0] * self.L
        self.phase_3_state = [0] * self.L
        self.phase_2_state = [0] * self.L

        for i in range(self.L):
            a = random.random()
            if a > self.gamma:
                self.Q_exe[i] = 0
            else:
                self.Q_exe[i] = 1
                self.num_tasks += 1
        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] == 1:
                self.vehicle_to_UAV_data[i] = random.randint(10, 20)
                self.RSU_computation_data[i] = random.randint(100, 200)
                self.UAV_to_RSU_data[i] = random.randint(30, 40)
                self.delay_constraint[i] = random.randint(100, 500)
                b = random.random()
                if 0 <= b <= 0.3:
                    # vehicle => UAV
                    self.phase_1[i] = 1
                    vehicle_to_UAV_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.phase_1_state[i] = vehicle_to_UAV_data_rest_ratio * self.vehicle_to_UAV_data[i]
                    self.phase_3_state[i] = 1 * self.RSU_computation_data[i]
                    self.phase_2_state[i] = 1 * self.UAV_to_RSU_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.9
                if 0.3 < b < 0.7:
                    # UAV => RSU
                    self.phase_2[i] = 1
                    self.phase_1_state[i] = 0
                    self.phase_3_state[i] = 1 * self.RSU_computation_data[i]
                    UAV_to_RSU_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.phase_2_state[i] = UAV_to_RSU_data_rest_ratio * self.UAV_to_RSU_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.6
                if 0.7 <= b <= 1:
                    # 计算阶段 in RSU
                    self.phase_3[i] = 1
                    self.phase_1_state[i] = 0
                    self.phase_2_state[i] = 0
                    computing_data_rest_ratio = 0.01 * random.randint(1, 100)
                    self.phase_3_state[i] = computing_data_rest_ratio * self.RSU_computation_data[i]
                    self.time_state[i] = self.delay_constraint[i] * 0.3
        state = self.phase_1_state + self.phase_2_state + self.phase_3_state + self.time_state
        return state

    def step(self, action):
        self.P_k = 30  # Tx power of vehicles: 30 dBm
        self.P_UAV = 27  # Tx power of RSU: 27 dBm

        # vehicle_to_UAV_B_ratio = softmax(np.array(action[:5])).tolist()
        # computing_resource_ratio = softmax(np.array(action[5:10])).tolist()
        # UAV_to_RSU_B_ratio = softmax(np.array(action[-5:])).tolist()

        vehicle_to_UAV_B_ratio = action[:5]
        computing_resource_ratio = action[5:10]
        UAV_to_RSU_B_ratio = action[-5:]

        reward = 0
        vehicle_to_UAV_rate = transmission_rate(len(action[:5]), self.P_k, vehicle_to_UAV_B_ratio, self.R_UAV)
        computing_rate = []
        for i in range(len(computing_resource_ratio)):
            computing_rate.append(computing_resource_ratio[i] * self.R_c * 1e3 / 1e3)
        UAV_to_RSU_rate = transmission_rate(len(action[:5]), self.P_UAV, UAV_to_RSU_B_ratio, self.R_RSU)


        reward_list = []
        for i in range(len(self.Q_exe)):
            if self.Q_exe[i] != 0:
                self.time_state[i] = self.time_state[i] - self.tao
                # vehicle_to_UAV_rate: Mbits/ms
                self.phase_1_state[i] -= self.phase_1[i] * self.tao * vehicle_to_UAV_rate[i]
                # computing_rate: Mcycles/ms
                self.phase_3_state[i] -= self.phase_3[i] * self.tao * computing_rate[i]
                self.phase_2_state[i] -= self.phase_2[i] * self.tao * UAV_to_RSU_rate[i]

                if self.time_state[i] <= 0 and self.phase_3_state[i] > 0:
                    self.Task_done[i] = -1
                    self.Q_exe[i] = 0
                    self.time_state[i] = 0
                    self.phase_1_state[i] = 0
                    self.phase_3_state[i] = 0
                    self.phase_2_state[i] = 0
                else:
                    if self.phase_1_state[i] <= 0:
                        self.phase_1_state[i] = 0
                        self.phase_1[i] = 0
                        self.phase_2[i] = 1
                    if self.phase_2_state[i] <= 0:
                        self.phase_2_state[i] = 0
                        self.phase_2[i] = 0
                        self.phase_3[i] = 1
                    if self.phase_3_state[i] <= 0:
                        self.phase_3_state[i] = 0
                        self.time_state[i] = 0
                        self.phase_3[i] = 0
                        self.Task_done[i] = 1
                        self.Q_exe[i] = 0
            reward += (self.Task_done[i] * 5 - (2 - self.time_state[i] / 500) * (self.phase_1_state[i] / 20 + self.phase_3_state[i] / 200 + self.phase_2_state[i] / 40) / 3)

        average_reward = reward / self.num_tasks

        new_state = self.phase_1_state + self.phase_2_state + self.phase_3_state + self.time_state
        count_complete = 0
        for i in range(len(self.Task_done)):
            if self.Task_done[i] == 1 or self.Task_done[i] == -1:
                count_complete += 1
        if count_complete == self.num_tasks:
            self.done = True
        else:
            self.done = False
        return new_state, average_reward, self.done
