import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Agent.Actor.Deterministic_Actor import Deterministic_Actor as Actor
from Agent.Critic.Q_Critic import Q_Critic as Critic
from Replay_Buffer.Replay_Buffer import Replay_Buffer, Prioritized_Replay_Buffer
from Utils.Common import clip_by_local_norm, update_target_model

tf.keras.backend.set_floatx('float32')


class TD3_Agent():
    def __init__(self, agent_index, state_shape, action_shape,
                 actor_unit_num_list: list = [32, 32], actor_activation="tanh", actor_lr=1e-3,
                 critic_unit_num_list: list = [32, 32, 32], critic_activation="linear", critic_lr=1e-3,
                 update_freq=1, actor_train_freq=2, gamma=0.98, tau=0.05, clip_norm=0.5,
                 batch_size=256, buffer_size=1e5, prioritized_replay=False,
                 alpha=0.6, beta=0.4, beta_increase=1e-3, min_priority=0.01, max_priority=1,
                 eval_noise_std=0.2, eval_noise_scale=1, eval_noise_bound=0.2
                 ):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.actor_unit_num_list = actor_unit_num_list
        self.actor_activation = actor_activation
        self.actor_lr = actor_lr
        self.critic_unit_num_list = critic_unit_num_list
        self.critic_activation = critic_activation
        self.critic_lr = critic_lr
        self.update_freq = update_freq
        self.actor_train_freq = actor_train_freq
        self.gamma = gamma
        self.tau = tau
        self.clip_norm = clip_norm
        self.train_step = 0

        self.eval_noise_std = eval_noise_std
        self.eval_noise_scale = eval_noise_scale
        self.eval_noise_bound = eval_noise_bound

        self.train_critic_1 = TD3_Critic(agent_index=self.agent_index,
                                         state_shape=self.state_shape, action_shape=self.action_shape,
                                         unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                         lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_1 = TD3_Critic(agent_index=self.agent_index,
                                          state_shape=self.state_shape, action_shape=self.action_shape,
                                          unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                          lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_1.model.set_weights(self.train_critic_1.model.get_weights())

        self.train_critic_2 = TD3_Critic(agent_index=self.agent_index,
                                         state_shape=self.state_shape, action_shape=self.action_shape,
                                         unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                         lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_2 = TD3_Critic(agent_index=self.agent_index,
                                          state_shape=self.state_shape, action_shape=self.action_shape,
                                          unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                          lr=self.critic_lr, clip_norm=self.clip_norm)
        self.target_critic_2.model.set_weights(self.train_critic_2.model.get_weights())

        self.train_actor_1 = TD3_Actor(agent_index=self.agent_index,
                                       state_shape=self.state_shape, action_shape=self.action_shape,
                                       unit_num_list=self.actor_unit_num_list, activation=self.actor_activation,
                                       lr=self.actor_lr, clip_norm=self.clip_norm, critic=self.train_critic_1)
        self.target_actor_1 = TD3_Actor(agent_index=self.agent_index,
                                        state_shape=self.state_shape, action_shape=self.action_shape,
                                        unit_num_list=self.actor_unit_num_list, activation=self.actor_activation,
                                        lr=self.actor_lr, clip_norm=self.clip_norm, critic=self.train_critic_1)
        self.target_actor_1.model.set_weights(self.train_actor_1.model.get_weights())

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.min_priority = min_priority
        self.max_priority = max_priority
        if self.prioritized_replay:
            self.replay_buffer = Prioritized_Replay_Buffer(buffer_size, self.alpha, self.beta, self.beta_increase,
                                                           self.min_priority, self.max_priority)
        else:
            self.replay_buffer = Replay_Buffer(buffer_size)

    def get_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        action_batch, log_prob_batch = self.train_actor_1.get_action(state_batch)
        return action_batch[0].numpy(), log_prob_batch[0].numpy()

    def get_target_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        action_batch, log_prob_batch = self.target_actor_1.get_action(state_batch)
        return action_batch[0].numpy(), log_prob_batch[0].numpy()

    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        self.replay_buffer.remember(state, action, log_prob, next_state, reward, done, dead)

    def train(self):
        self.train_step += 1
        if self.prioritized_replay:
            state_batch, action_batch, _, next_state_batch, reward_batch, _, dead_batch, index_batch, weight_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.expand_dims(tf.convert_to_tensor(weight_batch, dtype=tf.float32), 1)
        else:
            state_batch, action_batch, _, next_state_batch, reward_batch, _, dead_batch = self.replay_buffer.sample(
                self.batch_size)
            weight_batch = tf.ones(shape=(self.batch_size, 1), dtype=tf.float32)

        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        reward_batch = tf.expand_dims(tf.convert_to_tensor(reward_batch, dtype=tf.float32), 1)
        dead_batch = tf.expand_dims(tf.convert_to_tensor(dead_batch, dtype=tf.float32), 1)

        next_action_batch, _ = self.target_actor_1.get_action(next_state_batch, False)
        # eval_noise = np.random.normal(loc=0.0, scale=self.eval_noise_std, size=next_action_batch.shape)
        # clip_eval_noise = np.clip(eval_noise * self.eval_noise_scale, -1 * self.eval_noise_bound,
        #                           1 * self.eval_noise_bound)
        # next_action_batch = clip_eval_noise + next_action_batch
        next_q_batch_1 = self.target_critic_1.get_value(next_state_batch, next_action_batch)
        next_q_batch_2 = self.target_critic_2.get_value(next_state_batch, next_action_batch)
        next_q_batch = tf.stack([next_q_batch_1, next_q_batch_2])
        next_q_batch = tf.reduce_min(next_q_batch, axis=0)
        target_q_batch = tf.stop_gradient(reward_batch + self.gamma * next_q_batch * (1 - dead_batch))
        self.train_critic_1.loss, td_error_batch_1 = self.train_critic_1.train(state_batch, action_batch,
                                                                               target_q_batch, weight_batch)
        self.train_critic_2.loss, td_error_batch_2 = self.train_critic_2.train(state_batch, action_batch,
                                                                               target_q_batch, weight_batch)
        # train_actor_1延迟更新
        if self.train_step % self.actor_train_freq == 0:
            self.train_actor_1.loss = self.train_actor_1.train(state_batch)
        if self.train_step % self.update_freq == 0:
            self.model_update()
        if self.prioritized_replay:
            self.replay_buffer.batch_update(index_batch,
                                            np.sum((np.square(td_error_batch_1) + np.square(td_error_batch_2)) / 2,
                                                   axis=1))

    def model_update(self):
        # if self.train_step % (self.update_freq * self.actor_train_freq) == 0:  # todo 2023/9/19 ren change
        if self.train_step % self.update_freq * self.actor_train_freq == 0:
            update_target_model(self.train_actor_1.model, self.target_actor_1.model, self.tau)
        update_target_model(self.train_critic_1.model, self.target_critic_1.model, self.tau)
        update_target_model(self.train_critic_2.model, self.target_critic_2.model, self.tau)

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.target_actor_1.model.save_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
        self.target_critic_1.model.save_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        self.target_critic_2.model.save_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
        file_path = file_path + "/Agent_{}_train.log".format(self.agent_index)
        if not os.path.isfile(file_path):
            file = open(file_path, "w")
            file.write(
                "class_name:" + str("TD3") +
                "\nseed:" + str(seed) +
                "\nstate_shape:" + str(self.state_shape) +
                "\naction_shape:" + str(self.action_shape) +
                "\nactor_unit_num_list:" + str(self.actor_unit_num_list) +
                "\nactor_activation:" + str(self.actor_activation) +
                "\nactor_lr:" + str(self.actor_lr) +
                "\ncritic_unit_num_list:" + str(self.critic_unit_num_list) +
                "\ncritic_activation:" + str(self.critic_activation) +
                "\ncritic_lr:" + str(self.critic_lr) +
                "\nupdate_freq:" + str(self.update_freq) +
                "\nactor_train_freq:" + str(self.actor_train_freq) +
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
                "\nmax_priority:" + str(self.max_priority) +
                "\neval_noise_std:" + str(self.eval_noise_std) +
                "\neval_noise_scale:" + str(self.eval_noise_scale) +
                "\neval_noise_bound:" + str(self.eval_noise_bound)
            )

    def model_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.target_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
            self.target_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
            self.train_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(self.agent_index))
        else:
            self.target_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(agent_index))
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(agent_index))
            self.target_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))
            self.target_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))
            self.train_critic_2.model.load_weights(file_path + "/Agent_{}_Critic_2_model.h5".format(agent_index))


class TD3_Actor(Actor):
    def __init__(self, agent_index, state_shape, action_shape, unit_num_list, activation, lr, clip_norm, critic):
        super().__init__(state_shape=state_shape, action_shape=action_shape, unit_num_list=unit_num_list,
                         activation=activation)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.critic = critic
        self.opt = keras.optimizers.Adam(self.lr)
        self.loss = 0

    @tf.function
    def train(self, state_batch):
        with tf.GradientTape() as tape:
            new_action_batch, _ = self.get_action(state_batch, False)
            q_batch = self.critic.get_value(state_batch, new_action_batch)
            loss = -1 * tf.reduce_mean(q_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


class TD3_Critic(Critic):
    def __init__(self, agent_index, state_shape, action_shape, unit_num_list, activation, lr, clip_norm):
        super().__init__(state_shape=state_shape, action_shape=action_shape, value_shape=[1],
                         unit_num_list=unit_num_list, activation=activation)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.loss = 0

    @tf.function
    def train(self, state_batch, action_batch, target_q_batch, weight_batch):
        with tf.GradientTape() as tape:
            q_batch = self.get_value(state_batch, action_batch)
            td_error_batch = target_q_batch - q_batch
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(td_error_batch) * weight_batch, axis=1))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, td_error_batch


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Pendulum-v1").unwrapped
    agent = TD3_Agent(1, [3], [1])
    rewards_list = []
    sum_step = 0
    for each in range(500):
        rewards = 0
        step = 0
        state, _ = env.reset()
        done = False
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action * 2)
            step += 1
            sum_step += 1
            dead = done
            reward = (reward + 8) / 8
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
