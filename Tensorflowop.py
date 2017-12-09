import tensorflow as tf

# generate a 1x2 matrix, this op is set as a node
matrix1 = tf.constant([[3., 3.]])

# generate a 2x1 matrix,
matrix2 = tf.constant([[2.], [2.]])

# return value is the result of matrix
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()


print(sess.run(hello))
