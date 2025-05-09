import tensorflow as tf
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float32')


# 用于拟合V(s)值的Critic模型
# 实例化参数: 状态空间维度, 值空间维度, 网络结构, 输出端激活函数
class V_Critic():
    def __init__(self, state_shape, value_shape, unit_num_list, activation):
        self.state_shape = state_shape
        self.value_shape = value_shape
        self.unit_num_list = unit_num_list
        self.activation = activation
        self.model = self.model_create()

    def model_create(self):
        # 创建状态输入端
        self.state_input_layer = [
            keras.Input(shape=sum(self.state_shape), name="critic_state_input")
        ]
        # 创建输入链接层
        self.input_concat_layer = keras.layers.Concatenate(name="critic_input_concat")
        # 创建中间层
        self.hidden_layer_list = [
            keras.layers.Dense(self.unit_num_list[each], activation="relu", name="critic_hidden_{}".format(each)) for each in range(len(self.unit_num_list))
        ]
        # 创建值输出端
        self.value_output_layer = [
            keras.layers.Dense(sum(self.value_shape), activation=self.activation, name="critic_value_output")
        ]
        # 创建输出链接层
        self.output_concat_layer = keras.layers.Concatenate(name="critic_output_concat")
        # 链接各层
        x = self.input_concat_layer(self.state_input_layer)
        for hidden_layer in self.hidden_layer_list:
            x = hidden_layer(x)
        output_list = []
        output_list.append(self.value_output_layer[0](x))
        output = self.output_concat_layer(output_list)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer, outputs=output)
        return model

    # 向前传播
    @tf.function
    def get_value(self, state_batch):
        value_batch = self.model(state_batch)
        return value_batch


if __name__ == "__main__":
    import numpy as np
    # critic = V_Critic([35], [32, 32, 32], [1],  "linear")
    critic = V_Critic(state_shape=[35], value_shape=[1], unit_num_list=[32, 32, 32], activation="linear")
    state_batch = np.random.uniform(size=(128, 35))
    summary_writer = tf.summary.create_file_writer("Demo/V_Critic/")
    tf.summary.trace_on(graph=True, profiler=True)
    with summary_writer.as_default():
        critic.get_value(state_batch)
        tf.summary.trace_export(name="V Critic Model", step=0, profiler_outdir="Demo/V_Critic/")