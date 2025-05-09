import tensorflow as tf
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float32')


# 用于输出确定性策略的Actor模型
# 实例化参数: 状态空间维度，动作空间维度, 网络结构, 输出端激活函数
class Deterministic_Actor():
    def __init__(self, state_shape, action_shape, unit_num_list, activation):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.unit_num_list = unit_num_list
        self.activation = activation
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
        # 创建动作输出端
        self.action_output_layer_list = [
            keras.layers.Dense(shape, activation=self.activation, name="actor_action_output_{}".format(each)) for each, shape in enumerate(self.action_shape)
        ]
        # 创建输出链接层
        self.output_concat_layer = keras.layers.Concatenate(name="actor_output_concat")
        # 链接各层
        x = self.input_concat_layer(self.state_input_layer)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        output_list = []
        for action_output_layer in self.action_output_layer_list:
            output_list.append(action_output_layer(x))
        output = self.output_concat_layer(output_list)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer, outputs=output)
        return model

    # 向前传播
    @tf.function
    def get_action(self, state_batch, prob=True):
        action_batch = self.model(state_batch)
        if prob:
            log_prob_batch = tf.ones(shape=(len(action_batch),1))
        else:
            log_prob_batch = None
        return action_batch, log_prob_batch


if __name__ == "__main__":
    import numpy as np
    actor = Deterministic_Actor([35], [6, 6, 6], [32, 32, 32], "softmax")
    state_batch = np.random.uniform(size=(128, 35))
    summary_writer = tf.summary.create_file_writer("Demo/Deterministic_Actor/")
    tf.summary.trace_on(graph=True, profiler=True)
    with summary_writer.as_default():
        actor.get_action(state_batch)
        tf.summary.trace_export(name="Deterministic Actor Model", step=0, profiler_outdir="Demo/Deterministic_Actor/")