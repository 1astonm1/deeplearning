import tensorflow as tf
import numpy as np

# 神经层里常见的参数通常有weights、biases和激励函数。

# 输入值、输入的大小、输出的大小和激励函数，我们设定默认的激励函数是None。


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义权重参数 （在insize 和 outsize中随机生成）
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)   # 偏差参数
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  # 神经网络未激活的值 输入乘权重加偏差
    if activation_function is None:
        outputs = Wx_plus_b     # 没有激活函数下的输出
    else:
        outputs = activation_function(Wx_plus_b)    # 在有激活函数下的输出
    return outputs


# 创建数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
# 利用占位符定义我们所需的神经网络的输入 None表示给多少个sample都可以 1表示有一个特征
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 我们开始定义隐藏层,利用之前的add_layer()函数


def creat_nerual_network():
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.sigmoid)
    # 定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。
    prediction = add_layer(l1, 10, 1, activation_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    # 梯度下降优化训练 0.1代表最小误差

    return train_step, loss


def train():
    train_step, loss, init = creat_nerual_network()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    sess.close()


if __name__ == '__main__':
    creat_nerual_network()
    train()
