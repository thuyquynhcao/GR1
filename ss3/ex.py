# import tensorflow as tf
# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# f = x*x*y +2
# sess = tf.Session
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
#     print(result)
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))