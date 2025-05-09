import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float32')


# 用于输出随机策略的Actor模型
# 实例化参数: 状态空间维度，动作空间维度, 网络结构, 输出端激活函数, 最小熵, 最小方差, 最大方差
class Gaussian_Actor():
    def __init__(self, state_shape, action_shape, unit_num_list, activation, log_prob_epsilon, min_log_std, max_log_std):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.unit_num_list = unit_num_list
        self.activation = activation
        self.log_prob_epsilon = log_prob_epsilon
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.model = self.model_create()

    def model_create(self):
        # 创建状态输入端
        self.state_input_layer = [
            keras.Input(shape=sum(self.state_shape), name="actor_state_input")
        ]
        # 创建输入链接层
        self.input_concat_layer = keras.layers.Concatenate(name="actor_input_concat")
        # 创建中间层
        self.hidden_layer_list = [
            keras.layers.Dense(self.unit_num_list[each], activation="relu", name="actor_hidden_{}".format(each)) for each in range(len(self.unit_num_list))
        ]
        # 创建均值输出端
        self.mu_output_layer = [
            keras.layers.Dense(sum(self.action_shape), activation="linear", name="actor_mu_output")
        ]
        # 创建方差输出端
        self.log_std_output_layer = [
            keras.layers.Dense(sum(self.action_shape), activation="linear", name="actor_log_std_output")
        ]
        # 创建输出链接层
        self.mu_concat_layer = keras.layers.Concatenate(name="actor_mu_concat")
        self.log_std_concat_layer = keras.layers.Concatenate(name="actor_log_std_concat")
        # 链接各层
        x = self.input_concat_layer(self.state_input_layer)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        mu_list = []
        mu_list.append(self.mu_output_layer[0](x))
        mu = self.mu_concat_layer(mu_list)
        log_std_list = []
        log_std_list.append(self.log_std_output_layer[0](x))
        log_std = self.log_std_concat_layer(log_std_list)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer, outputs=[mu, log_std])
        return model

    # 向前传播
    @tf.function
    def get_action(self, state_batch, prob=True):
        mu_batch, log_std_batch = self.model(state_batch)
        log_std_batch = tf.clip_by_value(log_std_batch, clip_value_min=self.min_log_std, clip_value_max=self.max_log_std)
        dist_batch = tfp.distributions.Normal(mu_batch, tf.exp(log_std_batch))
        u_batch = dist_batch.sample()
        action_batch = tf.tanh(u_batch)
        if prob:
            log_prob_batch = tf.reduce_sum(dist_batch.log_prob(u_batch) - tf.math.log(1.0 - tf.pow(action_batch, 2) + self.log_prob_epsilon), axis=-1, keepdims=True)
        else:
            log_prob_batch = None
        return action_batch, log_prob_batch


if __name__ == "__main__":
    import numpy as np
    actor = Gaussian_Actor([35], [6, 6, 6], [32, 32, 32], "tanh", 1e-6, -20, 2)
    state_batch = np.random.uniform(size=(128, 35))
    summary_writer = tf.summary.create_file_writer("Demo/Gaussian_Actor/")
    tf.summary.trace_on(graph=True, profiler=True)
    with summary_writer.as_default():
        actor.get_action(state_batch)
        tf.summary.trace_export(name="Gaussian Actor Model", step=0, profiler_outdir="Demo/Gaussian_Actor/")