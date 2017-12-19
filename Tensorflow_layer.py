import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# define the layer of neural
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_puls_b'):
            Wx_puls_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_puls_b
        else:
            outputs = activation_function(Wx_puls_b)
            tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs


# make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hide layer
layer1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, 2, activation_function=None)

# define loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
# define the steo of train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# draw the coordinate system
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# tensorflow run
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    tf.global_variables_initializer().run()
    for step in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            rs = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(rs, step)
            try:
                lines = ax.lines
                lines.remove(lines[0])
                # ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.1)
    plt.pause(10)
