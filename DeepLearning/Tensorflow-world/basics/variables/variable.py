import tensorflow as tf

# 定义变量
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.), name="weights")
bias = tf.Variable(tf.zeros([3]), name="bias")
customs = tf.Variable(tf.zeros([0]), name="custom")

all_variables = [weights, bias, customs]

# 使用之前要初始化
init_customs = tf.variables_initializer(var_list=all_variables)

a = tf.multiply(weights, bias)

with tf.Session() as sess:
    sess.run(init_customs)
    result = sess.run(a)
    print(result)
    sess.close()
