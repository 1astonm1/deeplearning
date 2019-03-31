import numpy as np
import tensorflow as tf

# creat data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 1 + 0.3

# 定义变量
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
# 如何计算预测值
y = weights * x_data + biases

# loss function
loss = tf.reduce_mean(tf.square(y - y_data))

# 梯度下降优化器，定义learning rate
optimzer = tf.train.GradientDescentOptimizer(0.5)
# 训练目标是loss最小化
train = optimzer.minimize(loss)
# 初始化变量，即初始化 Weights 和 biases
init = tf.global_variables_initializer()
# 创建session，进行参数初始化
sess = tf.Session()
sess.run(init)

# 开始训练200步，每隔20步输出一下两个参数
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))

