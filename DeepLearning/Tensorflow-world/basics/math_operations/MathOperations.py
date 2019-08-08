import tensorflow as tf

# 定义常量 可选参数： value, dtype, shape, name
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# 基础运算
x = tf.add(a, b, name="add")        # x 代表a+b这个操作
y = tf.div(a, b, name="divide")     # y 代表a/b

# 运行 定义会话
with tf.Session() as sess:
    result = sess.run(x)       #  sess指向x运行
    writer = tf.summary.FileWriter("./logs", sess.graph)  # 写入tensorboard文件
    print(result)                                                                    # 在cmd中 tensorboard --logdir= “dir”
    writer.close()                                                                  # 浏览器中打开localhost:6006
    sess.close()    # 关闭会话



