import numpy as np


def creat_dataset():  # 通过numpy建立测试用数据集
    x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
    noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
    y_data = np.square(x_data) - 0.5 + noise
    return x_data, y_data


def read_data():    # 读取txt文件中的数据。最大的问题是通过python写入的数据中包含【】，不能直接读取数字
    x_data = []
    y_data = []
    fr = open("dataset.txt", "r")
    for line in fr.readlines():
        lineArr = line.replace('[', '')     # 去除数据集中的‘[’
        lineArr = lineArr.replace(']', '')  # 去除数据集中的‘]’
        lineArr = lineArr.strip().split()   # 将数据集从中间分开
        x_data.append(float(lineArr[0]))    # 分别读取对应的部分
        y_data.append(float(lineArr[1]))
    x_data = np.array(x_data).reshape(300, 1)   # 例如在tensorflow中对输入的形状有要求的话
    y_data = np.array(y_data).reshape(300, 1)   # 就reshape一下
    return x_data, y_data


def write_data(x_data, y_data):     # 写入数据集，两个数据写成两行
    with open("dataset.txt", "w") as f:
        for i in range(0, len(x_data)):
            f.write(str(x_data[i]))
            f.write('\t')
            f.write(str(y_data[i]))
            f.write('\n')


if __name__ == '__main__':
    # 创建和写入数据
    # x_data, y_data = creat_dataset()
    # write_data(x_data, y_data)
    # 读取数据
    x_data, y_data = read_data()
    print(x_data, y_data)


