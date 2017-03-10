import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

BS = 32
LR = 1e-4
ITER = 20000
VERBOSE = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
inp_images = tf.reshape(X, [-1,28,28,1])

# first conv layer
W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 16], stddev=0.1), name="W_conv1")
B_conv1 = tf.Variable(tf.random_normal([16], stddev=0.1), name="B_conv1")
conv1 = tf.nn.conv2d(inp_images, W_conv1, strides=[1,1,1,1], padding='SAME') + B_conv1

# first relu layer
relu1 = tf.nn.relu(conv1)

# first pooling layer
pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# second conv layer
W_conv2 = tf.Variable(tf.random_normal([5, 5, 16, 8], stddev=0.1), name="W_conv2")
B_conv2 = tf.Variable(tf.random_normal([8], stddev=0.1), name="B_conv2")
conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1,1,1,1], padding='SAME') + B_conv2

# second relu layer
relu2 = tf.nn.relu(conv2)

# second pooling layer
encoded = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# third conv layer
W_conv3 = tf.Variable(tf.random_normal([5, 5, 8, 8], stddev=0.1), name="W_conv3")
B_conv3 = tf.Variable(tf.random_normal([8], stddev=0.1), name="B_conv3")
conv3 = tf.nn.conv2d(encoded, W_conv3, strides=[1,1,1,1], padding='SAME') + B_conv3

relu3 = tf.nn.relu(conv3)

up1 = tf.image.resize_nearest_neighbor(relu3, [14, 14])

# fourth conv layer
W_conv4 = tf.Variable(tf.random_normal([5, 5, 8, 16], stddev=0.1), name="W_conv4")
B_conv4 = tf.Variable(tf.random_normal([16], stddev=0.1), name="B_conv4")
conv4 = tf.nn.conv2d(up1, W_conv4, strides=[1,1,1,1], padding='SAME') + B_conv4

relu4 = tf.nn.relu(conv4)

up2 = tf.image.resize_nearest_neighbor(relu4, [28, 28])


# decoding conv layer
W_dec = tf.Variable(tf.random_normal([5, 5, 16, 1], stddev=0.1), name="W_dec")
B_dec = tf.Variable(tf.random_normal([1], stddev=0.1), name="B_dec")
out_images = tf.nn.conv2d(up2, W_dec, strides=[1,1,1,1], padding='SAME') + B_dec

mseloss = tf.nn.l2_loss(out_images-inp_images)
train_step = tf.train.AdamOptimizer(LR).minimize(mseloss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(ITER):
  batch = mnist.train.next_batch(BS)
  if i%VERBOSE == 0:
    train_accuracy = mseloss.eval(feed_dict={X:batch[0]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={X: batch[0]})

print("test accuracy %g"%mseloss.eval(feed_dict={X: mnist.test.images}))
