import tensorflow as tf
import numpy as np
import skimage.io as io
import tflearn
from matplotlib import pyplot as plt
from tfsampler2 import ofsampler, batch_ofsampler

im = io.imread('Lenna.png')
im_ = (im.astype(np.float32)/255).tolist()

R = im[:,:,1]

ITER = 100000
VERBOSE = 20
LR = 0.001/6
DECAY = 1000

inp = tf.placeholder(dtype=tf.float32, shape=[R.shape[0], R.shape[1], 3])
out = tf.placeholder(dtype=tf.float32, shape=[R.shape[0], R.shape[1], 3])

W = im.shape[1]
H = im.shape[0]

Xgnp,Ygnp = np.meshgrid(np.arange(W), np.arange(H))

Xg = tf.constant(Xgnp.tolist(), dtype=tf.float32)
Yg = tf.constant(Ygnp.tolist(), dtype=tf.float32)

Vx = 5.0*tf.ones(R.shape, dtype=tf.float32)
Vy = 5.0*tf.ones(R.shape, dtype=tf.float32)




# build the graph
base = tf.constant(1, dtype=tf.float32, shape=[1,4,4,1], name='Const')

conv1 = tflearn.conv_2d(base, 8, [4, 4], activation='relu', name='conv1')
up8 = tf.image.resize_nearest_neighbor(conv1, [8,8])

conv2 = tflearn.conv_2d(up8, 8, [4, 4], activation='relu', name='conv2')
up16 = tf.image.resize_nearest_neighbor(conv2, [16,16])

conv3 = tflearn.conv_2d(up16, 8, [4, 4], activation='relu', name='conv3')
up32 = tf.image.resize_nearest_neighbor(conv3, [32,32])

conv4 = tflearn.conv_2d(up32, 8, [4, 4], activation='relu', name='conv4')
up64 = tf.image.resize_nearest_neighbor(conv4, [64,64])

conv5 = tflearn.conv_2d(up64, 8, [4, 4], activation='relu', name='conv5')
up128 = tf.image.resize_nearest_neighbor(conv5, [128,128])

conv6 = tflearn.conv_2d(up128, 8, [4, 4], activation='relu', name='conv6')
up256 = tf.image.resize_nearest_neighbor(conv6, [256,256])

conv7x = tflearn.conv_2d(up256, 8, [4, 4], activation='relu', name='conv7x')
up512x = tf.image.resize_nearest_neighbor(conv7x, [512,512])

conv7y = tflearn.conv_2d(up256, 8, [4, 4], activation='relu', name='conv7y')
up512y = tf.image.resize_nearest_neighbor(conv7y, [512,512])

Flowx = tf.squeeze(tflearn.conv_2d(up512x, 1, [4, 4], activation='relu', name='Flowx'))
Flowy = tf.squeeze(tflearn.conv_2d(up512y, 1, [4, 4], activation='relu', name='Flowy'))

out = ofsampler(inp, Vx, Vy, Xg, Yg)
im_nn_out = ofsampler(inp, Flowx, Flowy, Xg, Yg)
loss = tf.reduce_mean(tf.square((im_nn_out-out)))

optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

im_shifted = out.eval(feed_dict={inp: im_})
plt.imshow(im_)
plt.pause(2)
plt.imshow(im_shifted)
plt.pause(2)

for i in range(ITER):
  if i%VERBOSE == 0:
    train_accuracy = loss.eval(feed_dict={inp:im_, out:im_shifted})
    out_image = im_nn_out.eval(feed_dict={inp:im_, out:im_shifted})
    out_image1 = np.squeeze(np.array(out_image))
    plt.imshow(out_image1)
    plt.pause(0.0001)
    print("step %d, training accuracy %g"%(i, train_accuracy))
  optimizer.run(feed_dict={inp:im_, out:im_shifted})
  if i % DECAY == 0 and i is not 0:
   LR = 0.0001
