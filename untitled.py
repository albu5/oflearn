import tensorflow as tf
from PIL import Image
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

data1 = misc.imread('Lenna.png').astype(np.float32)/255
print (data1).shape
data = [data1.tolist()]
print data[0][0][0][0]

ITER = 100000
VERBOSE = 20
LR = 0.001


inp_img = tf.placeholder(dtype=tf.float32, shape=[1,512,512,3])

base = tf.constant(1, dtype=tf.float32, shape=[1,4,4,1], name='Const')

# first conv layer
W_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev=0.1), name="W_conv1")
B_conv1 = tf.Variable(tf.random_normal([32], stddev=0.1), name="B_conv1")
conv1 = tf.nn.conv2d(base, W_conv1, strides=[1,1,1,1], padding='SAME') + B_conv1

# first relu layer
relu1 = tf.nn.relu(conv1)

up1 = tf.image.resize_nearest_neighbor(relu1, [5, 5])


# second conv layer
W_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1), name="W_conv2")
B_conv2 = tf.Variable(tf.random_normal([64], stddev=0.1), name="B_conv2")
conv2 = tf.nn.conv2d(up1, W_conv2, strides=[1,1,1,1], padding='SAME') + B_conv2

# first relu layer
relu2 = tf.nn.relu(conv2)

up2 = tf.image.resize_nearest_neighbor(relu2, [16, 16])


# second conv layer
W_conv3 = tf.Variable(tf.random_normal([5, 5, 64, 32], stddev=0.1), name="W_conv3")
B_conv3 = tf.Variable(tf.random_normal([32], stddev=0.1), name="B_conv3")
conv3 = tf.nn.conv2d(up2, W_conv3, strides=[1,1,1,1], padding='SAME') + B_conv3

# first relu layer
relu3 = tf.nn.relu(conv3)

up3 = tf.image.resize_nearest_neighbor(relu3, [32, 32])


# second conv layer
W_conv4 = tf.Variable(tf.random_normal([5, 5, 32, 16], stddev=0.1), name="W_conv4")
B_conv4 = tf.Variable(tf.random_normal([16], stddev=0.1), name="B_conv4")
conv4 = tf.nn.conv2d(up3, W_conv4, strides=[1,1,1,1], padding='SAME') + B_conv4

# first relu layer
relu4 = tf.nn.relu(conv4)

up4 = tf.image.resize_nearest_neighbor(relu4, [64, 64])


# second conv layer
W_conv5 = tf.Variable(tf.random_normal([5, 5, 16, 8], stddev=0.1), name="W_conv5")
B_conv5 = tf.Variable(tf.random_normal([8], stddev=0.1), name="B_conv5")
conv5 = tf.nn.conv2d(up4, W_conv5, strides=[1,1,1,1], padding='SAME') + B_conv5

# first relu layer
relu5 = tf.nn.relu(conv5)

up5 = tf.image.resize_nearest_neighbor(relu5, [128,128])


# second conv layer
W_conv6 = tf.Variable(tf.random_normal([5, 5, 8, 4], stddev=0.1), name="W_conv6")
B_conv6 = tf.Variable(tf.random_normal([4], stddev=0.1), name="B_conv6")
conv6 = tf.nn.conv2d(up5, W_conv6, strides=[1,1,1,1], padding='SAME') + B_conv6

# first relu layer
relu6 = tf.nn.relu(conv6)

up6 = tf.image.resize_nearest_neighbor(relu6, [256,256])


# second conv layer
W_conv7 = tf.Variable(tf.random_normal([5, 5, 4, 3], stddev=0.1), name="W_conv7")
B_conv7 = tf.Variable(tf.random_normal([3], stddev=0.1), name="B_conv7")
conv7 = tf.nn.conv2d(up6, W_conv7, strides=[1,1,1,1], padding='SAME') + B_conv7

# first relu layer
relu7 = tf.nn.relu(conv7)

up7 = tf.image.resize_nearest_neighbor(relu7, [512,512])

err = up7-inp_img
loss = tf.reduce_sum(tf.square(err))


optmizer = tf.train.AdamOptimizer(LR).minimize(loss)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer().run()


for i in range(ITER):
  if i%VERBOSE == 0:
    train_accuracy = loss.eval(feed_dict={inp_img:data})
    out_image = up7.eval(feed_dict={inp_img:data})
    out_image1 = np.squeeze(np.array(out_image))
    plt.imshow(out_image1)
    plt.pause(0.5)
    print("step %d, training accuracy %g"%(i, train_accuracy))
  optmizer.run(feed_dict={inp_img: data})

print("test accuracy %g"%loss.eval(feed_dict={inp_img: data}))