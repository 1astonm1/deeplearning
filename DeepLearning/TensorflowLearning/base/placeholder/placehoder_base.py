import tensorflow as tf

# placeholder 是 Tensorflow 中的占位符，暂时储存变量.

# Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(),
# 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).

# 设定两个输入值：
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 设定输出值为两个数相乘
output = tf.multiply(input1, input2)

# 用session执行
sess = tf.Session()
result = sess.run(output, feed_dict={input1: [7.0], input2: [2.0]})
print(result)
sess.close()
