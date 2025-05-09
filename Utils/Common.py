import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

def clip_by_local_norm(gradients, norm):
    for idx, grad in enumerate(gradients):
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients

def batch_norm(batch, min_std=1e-5):
    batch = (batch - tf.reduce_mean(batch)) / (tf.math.reduce_std(batch) + min_std)
    return batch

def update_target_model(model: tf.keras.Model, target_model: tf.keras.Model, tau):
    model_weight = model.get_weights()
    target_model_weight = target_model.get_weights()
    new_model_weight = target_model_weight
    for i in range(len(new_model_weight)):
        new_model_weight[i] = tau * model_weight[i] + (1 - tau) * target_model_weight[i]
    target_model.set_weights(new_model_weight)

def discount_reward(reward_batch, done_batch, gamma):
    discount_reward_batch = []
    discount_reward = 0
    for reward, done in zip(reward_batch[::-1], done_batch[::-1]):
        if done:
            discount_reward = 0
        discount_reward = reward + gamma * discount_reward * (1 - done)
        discount_reward_batch.insert(0, discount_reward)
    discount_reward_batch = tf.stack(discount_reward_batch)
    return discount_reward_batch

def gae(td_error_batch, done_batch, gamma, lamba):
    gae_batch = []
    gae = 0
    for td_error, done in zip(td_error_batch[::-1], done_batch[::-1]):
        if done:
            gae = 0
        gae = td_error + gamma * lamba * gae * (1 - done)
        gae_batch.insert(0, gae)
    gae_batch = tf.stack(gae_batch)
    return gae_batch

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_gif(frames, file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
    patch = plt.imshow(frames[0])
    plt.axis("off")
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(fig=plt.gcf(), func=animate, frames=len(frames), interval=1)
    anim.save(file_path, writer="pillow", fps=30)