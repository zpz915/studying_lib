import tensorflow as tf





# def main():
#
#     #1.单变量函数
#     x = tf.Variable(initial_value = 3.)
#
#     with tf.GradientTape() as tape:
#         y = tf.square(x)
#
#     y_grad = tape.gradient(y,x)
#     print(y,y_grad)
#
#     #2.多元函数求导
#     x = tf.constant([[1.,2.],[3.,4.]])
#     y = tf.constant([[1.],[2.]])
#
#     w = tf.Variable(initial_value=[[1.],[2.]])
#     b = tf.Variable(initial_value=1.)
#
#     with tf.GradientTape() as tape:  #定义损失函数
#         L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x,w) + b -y))
#
#     w_grad,b_grad = tape.gradient(L,[w,b])
#     print(L.numpy(),w_grad.numpy(),b_grad.numpy())
#
#     #3.线性回归求解线性模型
#     import numpy as np
#
#     X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
#     y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
#
#     X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min()) #归一化处理
#     y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
#
#
#     #定义好数据类型以及参数类型
#     # 1、定义Tensor类型
#     X = tf.constant(X)
#     y = tf.constant(y)
#
#     # 2、参数使用变量初始化
#     a = tf.Variable(initial_value=0.)
#     b = tf.Variable(initial_value=0.)
#     variables = [a, b]
#
#     num_epoch = 100
#     optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
#
#     for e in range(num_epoch):
#         # 使用tf.GradientTape()记录损失函数的梯度信息
#         with tf.GradientTape() as tape:
#             y_pred = a * X + b
#             loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
#         # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
#         grads = tape.gradient(loss, variables)
#         # TensorFlow自动根据梯度更新参数
#         optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
#
#     print(a.numpy(), b.numpy())
#
#


class Linear(tf.keras.Model):
    """
    自定义线性回归层
    """
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation = None,
            kernel_initializer = tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self,input):
        output = self.dense(input)
        return output

def linear_with_model():
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    #定义训练过程
    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = 0.5 * tf.reduce_mean(tf.square(y_pred - y))

        #梯度计算
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    print(model.variables)
    print(model.layers)

    print(model.summary())
    print(model.inputs)
    print(model.outputs)

if __name__ == '__main__':
    linear_with_model()