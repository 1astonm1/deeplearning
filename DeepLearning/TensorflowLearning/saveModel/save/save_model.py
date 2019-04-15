import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 定义权重参数 （在insize 和 outsize中随机生成）
    biases = tf.Variable(tf.zeros([1, out_size])+0.1)   # 偏差参数
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)  # 神经网络未激活的值 输入乘权重加偏差
    if activation_function is None:
        outputs = Wx_plus_b     # 没有激活函数下的输出
    else:
        outputs = activation_function(Wx_plus_b)    # 在有激活函数下的输出
    return outputs


def creat_nerual_network_4():
    l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    l2 = add_layer(l1, 10, 10, activation_function=tf.nn.relu)
    l3 = add_layer(l2, 10, 10, activation_function=tf.nn.relu)
    l4 = add_layer(l3, 10, 5, activation_function=tf.nn.relu)
    prediction = add_layer(l4, 5, 1, activation_function=None)
    # 定义输出层。此时的输入就是隐藏层的输出——l1，输入有10层（隐藏层的输出层），输出有1层。
    # l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
    # prediction = add_layer(l1, 10, 1, activation_function=None)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    # 计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
    # 梯度下降优化训练 0.1代表最小误差

    return train_step, loss, prediction

def display_dataset(x_data, y_data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_data, y_data)
    plt.ion()  # 本次运行请注释，全局运行不要注释
    plt.show()
    return ax


def train_and_save(xs, x_data, ys, y_data, ax):
    train_step, loss, prediction = creat_nerual_network_4()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 500 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(x_data, y_data)
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            # plot the prediction
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
            plt.show()

    saver = tf.train.Saver()  # 使用tf.train.saver()保存模型
    saver.save(sess, "./model/MyModel!")
    sess.close()

####################################################################################################
# 其他例子：
# saver.save(sess, './checkpoint_dir/MyModel',global_step=1000)  1000次迭代后保存模型
# saver.save(sess, './checkpoint_dir/MyModel',global_step=step,write_meta_graph=False)  指定不保存图
# tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)  每两小时保存一次，只保存最近的5个

# 注意！文件名一样的话就会覆盖保存，不一样的话不会影响前面的保存模型
#####################################################################################################


if __name__ == '__main__':
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])
    ax = display_dataset(x_data, y_data)
    train_and_save(xs, x_data, ys, y_data, ax)
