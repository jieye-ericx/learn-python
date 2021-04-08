import tensorflow as tf


# 线性回归
def myregression():
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

    y_true = tf.matmul(x, [[0.7]]) + 0.8
    weight = tf.Variable(tf.random_normal([1, 1], mean=0.0, stddev=1.0, name="w"))
    bias = tf.Variable(0.0, name="b")

    y_predict = tf.matmul(x, weight) + bias

    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        print("随即初始化参数权重：%f，偏置：%f" % (weight.eval(), bias.eval()))

        for i in range(100):
            sess.run(train_op)
            print("第%d次优化参数权重为：%f，偏置为:%f" % (i, weight.eval(), bias.eval()))
    return None

if __name__ == "__main__":
    myregression()
