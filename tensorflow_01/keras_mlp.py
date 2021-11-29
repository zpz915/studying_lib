import tensorflow as tf

import numpy as np


num_epochs = 5
batch_size = 64
learning_rate = 0.001

class MnistLoader(object):
  """

  数据加载处理类
  """
  def __init__(self):
    #加载数据
    (self.train_data, self.train_label), (self.test_data, self.test_label) = \
        tf.keras.datasets.mnist.load_data()

    #2.处理数据  归一化 维度拓展 类型
    self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0,-1)
    self.test_data =np.expand_dims(self.test_data.astype(np.float32)/255.0,-1)


    self.train_label = self.train_label.astype(np.int32)
    self.test_label = self.test_label.astype(np.int32)

    #获取一个变量接受数据量
    self.num_train_data,self.num_test_data = self.train_data.shape[0],self.test_data.shape[0]

  def get_batch(self,batch_size):
    """按照序列获得指定数据大小的批次数据

    :param batch_size:每批次数据的大小
    :return:
    """
    #获取随机生成的batch_size大小的数据的下标  乱序操作
    index = np.random.randint(0,self.train_data.shape[0],batch_size)

    return self.train_data[index,:],self.train_label[index]


class MLP(tf.keras.Model):
    """
    自定义MLP类
    """
    def __init__(self):
        super().__init__()
        #卷积到全联接层的数据形状处理
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)


    def call(self,inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)

        output = tf.nn.softmax(x)
        return output

def train():
    """
    模型训练逻辑







    :return:
    """
    # 1、从 DataLoader 中随机取一批训练数据；并初始化模型
    mnist = MnistLoader()
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 2、将这批数据送入模型，计算出模型的预测值；
    #总共样本len(train_Data)  迭代次数epochs表示数据过几遍  batch_size 每次批次训练的样本
    #一共需要多少批次
    num_batches = int(mnist.num_train_data//batch_size *num_epochs)
    for batch_index in range(num_batches):
        x,y = mnist.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x)
            #3、将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=y,y_pred=y_pred))
        #4、计算损失函数关于模型变量的导数；
        grads = tape.gradient(loss,model.variables)
        # 5、将求出的导数值传入优化器，使用优化器的apply_gradients方法更新模型参数以最小化损失函数（优化器的详细使用方法见前章 ）。

        optimizer.apply_gradients(grads_and_vars =zip(grads,model.variables))

    #三、对测试数据集进行评估
    y_pred = model.predict(mnist.test_data)
    #初始化一个metrics
    a = tf.keras.metrics.SparseCategoricalAccuracy()
    a.update_state(y_true=mnist.test_label, y_pred=y_pred)
    print("测试准确率: %f" % a.result())



if __name__ == '__main__':
    # mnist = MnistLoader()
    # train_data,train_label = mnist.get_batch(64)
    # print(train_data,train_label)
    train()