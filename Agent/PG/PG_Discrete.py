import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from Agent.Actor.Deterministic_Actor import Deterministic_Actor as Actor
from Replay_Buffer.Replay_Buffer import Replay_Buffer
from Utils.Common import clip_by_local_norm, batch_norm, discount_reward

tf.keras.backend.set_floatx("float32")


class PG_Agent():
    def __init__(self, agent_index, state_shape, action_shape,
                 actor_unit_num_list:list=[32, 32], actor_activation="softmax", actor_lr=1e-3,
                 gamma=0.98, clip_norm=0.5, buffer_size=1e5):
        self.agent_index = agent_index
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.actor_unit_num_list = actor_unit_num_list
        self.actor_activation = actor_activation
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.clip_norm = clip_norm
        self.train_step = 0

        self.train_actor_1 = PG_Actor(agent_index=self.agent_index,
                                      state_shape=self.state_shape, action_shape=self.action_shape,
                                      unit_num_list=self.actor_unit_num_list, activation=self.actor_activation,
                                      lr=self.actor_lr, clip_norm=self.clip_norm)

        self.buffer_size = buffer_size
        self.replay_buffer = Replay_Buffer(buffer_size)

    def get_action(self, state):
        state_batch = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        prob_batch, _ = self.train_actor_1.get_action(state_batch)
        dist_batch = tfp.distributions.Categorical(prob_batch)
        action_batch = dist_batch.sample()
        log_prob_batch = dist_batch.log_prob(action_batch)
        return action_batch[0].numpy(), log_prob_batch[0].numpy()

    def remember(self, state, action, log_prob, next_state, reward, done, dead):
        self.replay_buffer.remember(state, np.array([action]), np.array([log_prob]), next_state, reward, done, dead)

    def train(self):
        self.train_step += 1
        state_batch, action_batch, _, _, reward_batch, done_batch, _ = map(np.asarray, zip(*self.replay_buffer.buffer))
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.int32)
        reward_batch = tf.expand_dims(tf.convert_to_tensor(reward_batch, dtype=tf.float32), 1)
        done_batch = tf.expand_dims(tf.convert_to_tensor(done_batch, dtype=tf.float32), 1)

        discount_reward_batch = tf.stop_gradient(batch_norm(discount_reward(reward_batch, done_batch, self.gamma)))
        self.train_actor_1.loss = self.train_actor_1.train(state_batch, action_batch, discount_reward_batch)
        self.replay_buffer.buffer.clear()

    def model_update(self):
        pass

    def model_save(self, file_path, seed):
        if os.path.exists(file_path):
            pass
        else:
            os.makedirs(file_path)
        self.train_actor_1.model.save_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
        file_path = file_path + "/Agent_{}_train.log".format(self.agent_index)
        if not os.path.isfile(file_path):
            file = open(file_path, "w")
            file.write(
                "class_name:" + str("PG") +
                "\nseed:" + str(seed) +
                "\nstate_shape:" + str(self.state_shape) +
                "\naction_shape:" + str(self.action_shape) +
                "\nactor_unit_num_list:" + str(self.actor_unit_num_list) +
                "\nactor_activation:" + str(self.actor_activation) +
                "\nactor_lr:" + str(self.actor_lr) +
                "\ngamma:" + str(self.gamma) +
                "\nclip_norm:" + str(self.clip_norm) +
                "\nbuffer_size:" + str(self.buffer_size)
            )

    def model_load(self, file_path, agent_index=None):
        if agent_index == None:
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(self.agent_index))
        else:
            self.train_actor_1.model.load_weights(file_path + "/Agent_{}_Actor_1_model.h5".format(agent_index))


class PG_Actor(Actor):
    def __init__(self, agent_index, state_shape, action_shape, unit_num_list, activation, lr, clip_norm):
        super().__init__(state_shape=state_shape, action_shape=action_shape, unit_num_list=unit_num_list, activation=activation)
        self.agent_index = agent_index
        self.lr = lr
        self.clip_norm = clip_norm
        self.opt = keras.optimizers.Adam(self.lr)
        self.entropy = 0
        self.loss = 0

    @tf.function
    def train(self, state_batch, action_batch, discount_reward_batch):
        with tf.GradientTape() as tape:
            prob_batch, _ = self.get_action(state_batch, False)
            dist_batch = tfp.distributions.Categorical(prob_batch)
            log_prob_batch = dist_batch.log_prob(tf.squeeze(action_batch))
            loss = -1 * tf.reduce_mean(log_prob_batch * tf.squeeze(discount_reward_batch))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients = clip_by_local_norm(gradients, self.clip_norm)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss


if __name__ == "__main__":
    import gym
    import matplotlib.pyplot as plt
    env = gym.make("CartPole-v1").unwrapped
    agent = PG_Agent(1, [4], [2])
    rewards_list = []
    sum_step = 0
    for each in range(2000):
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
        agent.train()
        rewards_list.append(rewards)
        print("Episode:", each, "Step", step, "Reward:", rewards, "Max Reward:", max(rewards_list))
    plt.plot(rewards_list)
    plt.show()