import tensorflow as tf

hello = tf.constant("Hello,TensorFlow")
a = tf.constant(10)
b = tf.constant(20)

sess = tf.Session()
print(sess.run(hello))
print(sess.run(a + b))
sess.close()
