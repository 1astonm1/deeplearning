import numpy as np
import time

# 此文件对比说明了向量化处理的作用，对比普通处理来说快很多

a = np.array([1, 2, 3, 4])
print(a)


a = np.random.rand(1000000)
b = np.random.rand(1000000)
tic = time.time()

# 向量化版本
c = np.dot(a, b)
toc = time.time()
print(c)
print(str((toc-tic)*1000))

# 结果：
# 249814.69017964465
# 0.0

# 普通计算版本
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
    toc = time.time()
print(c)
print(str(1000*(toc-tic)))

# 249814.69017963848
# 830.416202545166
