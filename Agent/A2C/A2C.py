import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from Agent.Actor.Gaussian_Actor import Gaussian_Actor as Actor
from Agent.Critic.V_Critic import V_Critic as Critic
from Replay_Buffer.Replay_Buffer import Replay_Buffer
from Utils.Common import clip_by_local_norm, batch_norm, gae

tf.keras.backend.set_floatx("float32")


class A2C_Agent():
    def __init__(self, agent_index, state_shape, action_shape,
                 actor_unit_num_list:list=[32, 32], actor_activation="tanh", actor_lr=1e-3,
                 critic_unit_num_list:list=[32, 32, 32], critic_activation="linear", critic_lr=1e-3,
                 gamma=0.98, clip_norm=0.5, buffer_size=1e5, lamba=0.95, log_prob_epsilon=1e-6, min_log_std=-20, max_log_std=2
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
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.train_step = 0

        self.lamba = lamba
        self.log_prob_epsilon = log_prob_epsilon
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.train_actor_1 = A2C_Actor(agent_index=self.agent_index,
                                       state_shape=self.state_shape, action_shape=self.action_shape,
                                       unit_num_list=self.actor_unit_num_list, activation=self.actor_activation,
                                       lr=self.actor_lr, clip_norm=self.clip_norm, log_prob_epsilon=self.log_prob_epsilon,
                                       min_log_std=self.min_log_std, max_log_std=self.max_log_std)

        self.train_critic_1 = A2C_Critic(agent_index=self.agent_index,
                                         state_shape=self.state_shape,
                                         unit_num_list=self.critic_unit_num_list, activation=self.critic_activation,
                                         lr=self.critic_lr, clip_norm=self.clip_norm)

        self.buffer_size = buffer_size
        self.replay_buffer = Replay_Buffer(buffer_size)

    def get_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        action_batch, log_prob_batch = self.train_actor_1.get_action(state_batch)
        return action_batch[0].numpy(), log_prob_batch[0].numpy()

    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        self.replay_buffer.remember(state, action, log_prob, next_state, reward, done, dead)

    def train(self):
        self.train_step += 1
        state_batch, action_batch, _, next_state_batch, reward_batch, done_batch, dead_batch = map(np.asarray, zip(*self.replay_buffer.buffer))
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        reward_batch = tf.expand_dims(tf.convert_to_tensor(reward_batch, dtype=tf.float32), 1)
        done_batch = tf.expand_dims(tf.convert_to_tensor(done_batch, dtype=tf.float32), 1)
        dead_batch = tf.expand_dims(tf.convert_to_tensor(dead_batch, dtype=tf.float32), 1)

        v_batch = self.train_critic_1.get_value(state_batch)
        next_v_batch = self.train_critic_1.get_value(next_state_batch)
        target_v_batch = tf.stop_gradient(reward_batch + self.gamma * next_v_batch * (1 - dead_batch))
        td_error_batch = target_v_batch - v_batch
        advantage_batch = tf.stop_gradient(batch_norm(gae(td_error_batch, done_batch, self.gamma, self.lamba)))
        self.train_critic_1.loss = self.train_critic_1.train(state_batch, target_v_batch)
        self.train_actor_1.loss = self.train_actor_1.train(state_batch, action_batch, advantage_batch)
        self.replay_buffer.buffer.clear()

    def model_update(self):
        pass

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.train_actor_1.model.save_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
        self.train_critic_1.model.save_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        file_path = file_path + "/Agent_{}_train.log".format(self.agent_index)
        if not os.path.isfile(file_path):
            file = open(file_path, "w")
            file.write(
                "class_name:" + str("A2C") +
                "\nseed:" + str(seed) +
                "\nstate_shape:" + str(self.state_shape) +
                "\naction_shape:" + str(self.action_shape) +
                "\nactor_unit_num_list:" + str(self.actor_unit_num_list) +
                "\nactor_activation:" + str(self.actor_activation) +
                "\nactor_lr:" + str(self.actor_lr) +
                "\ncritic_unit_num_list:" + str(self.critic_unit_num_list) +
                "\ncritic_activation:" + str(self.critic_activation) +
                "\ncritic_lr:" + str(self.critic_lr) +
                "\ngamma:" + str(self.gamma) +
                "\nclip_norm:" + str(self.clip_norm) +
                "\nbuffer_size:" + str(self.buffer_size) +
                "\nlambda:" + str(self.lamba) +
                "\nlog_prob_epsilon:" + str(self.log_prob_epsilon) +
                "\nmin_log_std:" + str(self.min_log_std) +
                "\nmax_log_std:" + str(self.max_log_std)
            )

    def model_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(self.agent_index))
        else:
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(agent_index))
            self.train_critic_1.model.load_weights(file_path + "/Agent_{}_Critic_1_model.h5".format(agent_index))


class A2C_Actor(Actor):
    def __init__(self, agent_index, state_shape, action_shape, unit_num_list, activation, lr, clip_norm,
                 log_prob_epsilon, min_log_std, max_log_std):
        super().__init__(state_shape=state_shape, action_shape=action_shape, unit_num_list=unit_num_list, activation=activation,
                         log_prob_epsilon=log_prob_epsilon, min_log_std=min_log_std, max_log_std=max_log_std)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.entropy = 0
        self.loss = 0

    @tf.function
    def get_action(self, state_batch, prob=True):
        mu_batch, log_std_batch = self.model(state_batch)
        log_std_batch = tf.clip_by_value(log_std_batch, clip_value_min=self.min_log_std, clip_value_max=self.max_log_std)
        dist_batch = tfp.distributions.Normal(mu_batch, tf.math.exp(log_std_batch))
        u_batch = tf.clip_by_value(dist_batch.sample(), -5, 5)
        action_batch = tf.math.tanh(u_batch)
        if prob:
            log_prob_batch = tf.reduce_sum(dist_batch.log_prob(u_batch) - tf.math.log(1.0 - tf.pow(action_batch, 2) + self.log_prob_epsilon), axis=-1, keepdims=True)
        else:
            log_prob_batch = None
        return action_batch, log_prob_batch

    @tf.function
    def train(self, state_batch, action_batch, advantage_batch):
        with tf.GradientTape() as tape:
            mu_batch, log_std_batch = self.model(state_batch)
            log_std_batch = tf.clip_by_value(log_std_batch, clip_value_min=self.min_log_std, clip_value_max=self.max_log_std)
            dist_batch = tfp.distributions.Normal(mu_batch, tf.math.exp(log_std_batch))
            u_batch = tf.math.atanh(action_batch)
            log_prob_batch = tf.reduce_sum(dist_batch.log_prob(u_batch) - tf.math.log(1.0 - tf.pow(action_batch, 2) + self.log_prob_epsilon), axis=-1, keepdims=True)
            loss = -1 * tf.reduce_mean(log_prob_batch * advantage_batch)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


class A2C_Critic(Critic):
    def __init__(self, agent_index, state_shape, unit_num_list, activation, lr, clip_norm):
        super().__init__(state_shape=state_shape, unit_num_list=unit_num_list, value_shape=[1], activation=activation)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.loss = 0

    @tf.function
    def train(self, state_batch, target_v_batch):
        with tf.GradientTape() as tape:
            v_batch = self.get_value(state_batch)
            loss = tf.reduce_mean(tf.square(target_v_batch - v_batch))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    env = gym.make("Pendulum-v1").unwrapped
    agent = A2C_Agent(1, [3], [1])
    rewards_list = []
    sum_step = 0
    for each in range(2000):
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
        agent.train()
        rewards_list.append(rewards)
        print("Episode:", each, "Step", step, "Reward:", rewards, "Max Reward:", max(rewards_list))
    plt.plot(rewards_list)
    plt.show()