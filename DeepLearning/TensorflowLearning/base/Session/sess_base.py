import tensorflow as tf
# Tensorflow的Session,对话控制模块，可以用session.run来运行框架中的某一个点的功能
# session.run(x) 其中x是需要运行的步骤

# 定义两个矩阵
matrix1 = tf.constant([[3, 2], [3, 4]])
matrix2 = tf.constant([[2, 4], [4, 5]])

# 将两个矩阵相乘：
product = tf.matmul(matrix1, matrix2)

# 使用sess运行相乘的步骤：
sess = tf.Session()
print(sess.run(product))
sess.close()

