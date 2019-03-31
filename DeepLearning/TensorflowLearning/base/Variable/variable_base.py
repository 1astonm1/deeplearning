# variable 是在tensorflow中使用的变量， 注意使用之前必须初始化

import tensorflow as tf
# 定义变量state,给定初始值和name
state = tf.Variable(0, name="counter")

# 可以直接输出参数
print(state.name)

# 使用变量
one = tf.constant(1)
new_value = tf.add(one, state)
update = tf.assign(state, new_value)

# 初始化变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 一定要记得init需要用sess run一下
for i in range(3):
    result = sess.run(update)
    print(result)
sess.close()
