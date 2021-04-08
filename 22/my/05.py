import tensorflow as tf

# 模拟同步处理数据
# 1。定义队列
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 2。放入数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3],])
#
# out_q = Q.dequeue()
#
# data = out_q + 1
# en_q = Q.enqueue(data)
#
# with tf.Session() as sess:
#     #     初始化队列u
#     sess.run(enq_many)
#
#     for i in range(100):
#         sess.run(en_q)
#
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))

# 模拟异步



