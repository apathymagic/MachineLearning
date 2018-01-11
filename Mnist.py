import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def compute_accuracy(sess, v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
# layer1 = add_layer(xs, 784, 500, tf.nn.relu)
prediction = add_layer(xs, 784, 10, tf.nn.softmax)

# calculate loss
# cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)), reduction_indices=[1])

# optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# run
with tf.session() as sess:
    tf.initialize_all_variables().run()
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if step % 50 == 0:
            print(compute_accuracy(sess, mnist.test.images, mnist.test.labels))
