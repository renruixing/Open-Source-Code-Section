import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from Agent.Critic.V_Critic import V_Critic as Critic
from Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer
from Utils.Common import clip_by_local_norm, update_target_model

tf.keras.backend.set_floatx('float32')


class DDQN_Agent():
    def __init__(self, agent_index, state_shape, action_shape,
                 critic_unit_num_list:list=[32, 32, 32], critic_activation="linear", critic_lr=1e-3,
                 update_freq=1, gamma=0.98, tau=0.05, clip_norm=0.5,
                 batch_size=256, buffer_size=1e5, prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, min_priority=0.01, max_priority=1
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.critic_unit_num_list = critic_unit_num_list
        self.critic_activation = critic_activation
        self.critic_lr = critic_lr
        self.update_freq = update_freq
        self.gamma = gamma
        self.tau = tau
        self.clip_norm = clip_norm
        self.train_step = 0

        self.train_critic_1 = DDQN_Critic(agent_index=self.agent_index,
                                          state_shape=self.state_shape, action_shape=self.action_shape,
                                          unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                          lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_1 = DDQN_Critic(agent_index=self.agent_index,
                                           state_shape=self.state_shape, action_shape=self.action_shape,
                                           unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                           lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_1.model.set_weights(self.train_critic_1.model.get_weights())

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.min_priority = min_priority
        self.max_priority = max_priority
        if self.prioritized_replay:
            self.replay_buffer = Prioritized_Replay_Buffer(buffer_size, self.alpha, self.beta, self.beta_increase, self.min_priority, self.max_priority)
        else:
            self.replay_buffer = Replay_Buffer(buffer_size)

    def get_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        q_batch = self.train_critic_1.get_value(state_batch)
        dist_batch = tfp.distributions.Categorical(q_batch)
        action_batch = dist_batch.sample()
        log_prob_batch = dist_batch.log_prob(action_batch)
        return action_batch[0].numpy(), log_prob_batch[0].numpy()

    def get_target_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        q_batch = self.target_critic_1.get_value(state_batch)
        dist_batch = tfp.distributions.Categorical(q_batch)
        action_batch = dist_batch.sample()
        log_prob_batch = dist_batch.log_prob(action_batch)
        return action_batch.numpy(), log_prob_batch.numpy()

    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        self.replay_buffer.remember(state, np.array([action]), np.array([log_prob]), next_state, reward, done, dead)

    def train(self):
        self.train_step += 1
        if self.prioritized_replay:
            state_batch, action_batch, _, next_state_batch, reward_batch, _, dead_batch, index_batch, weight_batch = self.replay_buffer.sample(self.batch_size)
            weight_batch = tf.expand_dims(tf.convert_to_tensor(weight_batch, dtype=tf.float32), 1)
        else:
            state_batch, action_batch, _, next_state_batch, reward_batch, _, dead_batch = self.replay_buffer.sample(self.batch_size)
            weight_batch = tf.ones(shape=(self.batch_size, 1), dtype=tf.float32)
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.int32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        reward_batch = tf.expand_dims(tf.convert_to_tensor(reward_batch, dtype=tf.float32), 1)
        dead_batch = tf.expand_dims(tf.convert_to_tensor(dead_batch, dtype=tf.float32), 1)

        q_batch = self.train_critic_1.get_value(state_batch)
        next_action_batch = tf.expand_dims(tf.math.argmax(tf.math.softmax(self.train_critic_1.get_value(next_state_batch)), axis=1), 1)
        next_q_batch = self.target_critic_1.get_value(next_state_batch)
        target_q_batch = q_batch.numpy()
        next_q_batch = next_q_batch.numpy()
        for each in range(self.batch_size):
            target_q_batch[each][action_batch[each]] = reward_batch[each] + self.gamma * next_q_batch[each][next_action_batch[each]] * (1 - dead_batch[each])
        target_q_batch = tf.stop_gradient(target_q_batch)
        self.train_critic_1.loss, td_error_batch = self.train_critic_1.train(state_batch, target_q_batch, weight_batch)
        if self.train_step % self.update_freq == 0:
            self.model_update()
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch, np.sum(np.square(td_error_batch), axis=1))

    def model_update(self):
        update_target_model(self.train_critic_1.model, self.target_critic_1.model, self.tau)

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_critic_1.model.save_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        file_path = file_path + "/Agent_{}_train.log".format(self.agent_index)
        if not os.path.isfile(file_path):
            file = open(file_path, "w")
            file.write(
                "class_name:" + str("DDQN") +
                "\nseed:" + str(seed) +
                "\nstate_shape:" + str(self.state_shape) +
                "\naction_shape:" + str(self.action_shape) +
                "\ncritic_unit_num_list:" + str(self.critic_unit_num_list) +
                "\ncritic_activation:" + str(self.critic_activation) +
                "\ncritic_lr:" + str(self.critic_lr) +
                "\nupdate_freq:" + str(self.update_freq) +
                "\ngamma:" + str(self.gamma) +
                "\ntau:" + str(self.tau) +
                "\nclip_norm:" + str(self.clip_norm) +
                "\nbatch_size:" + str(self.batch_size) +
                "\nbuffer_size:" + str(self.buffer_size) +
                "\nPER:" + str(self.prioritized_replay) +
                "\nalpha:" + str(self.alpha) +
                "\nbeta:" + str(self.beta) +
                "\nbeta_increase:" + str(self.beta_increase) +
                "\nmin_priority:" + str(self.min_priority) +
                "\nmax_priority:" + str(self.max_priority)
            )

    def model_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        else:
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))


class DDQN_Critic(Critic):
    def __init__(self, agent_index, state_shape, action_shape, unit_num_list, activation, lr, clip_norm):
        super().__init__(state_shape=state_shape, value_shape=action_shape, unit_num_list=unit_num_list, activation=activation)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.loss = 0

    @tf.function
    def train(self, state_batch, target_q_batch, weight_batch):
        with tf.GradientTape() as tape:
            q_batch = self.get_value(state_batch)
            td_error_batch = target_q_batch - q_batch
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(td_error_batch) * weight_batch, axis=1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, td_error_batch


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    env = gym.make("CartPole-v1").unwrapped
    agent = DDQN_Agent(1, [4], [2])
    rewards_list = []
    sum_step = 0
    for each in range(500):
        rewards = 0
        step = 0
        state, _ = env.reset()
        done = False
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            sum_step += 1
            step += 1
            dead = done
            if step >= 200:
                done = True
            agent.remember(state, action, log_prob, next_state, reward, done, dead)
            rewards += reward
            state = next_state
            if sum_step % 10 == 0 and agent.replay_buffer.size() >= agent.batch_size * 10:
                agent.train()
        rewards_list.append(rewards)
        print("Episode:", each, "Step", step, "Reward:", rewards, "Max Reward:", max(rewards_list))
    plt.plot(rewards_list)
    plt.show()