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
LR = 0.01
DECAY = 1000

inp = tf.placeholder(dtype=tf.float32, shape=[R.shape[0], R.shape[1], 3])
out = tf.placeholder(dtype=tf.float32, shape=[R.shape[0], R.shape[1], 3])

W = im.shape[1]
H = im.shape[0]

Xgnp,Ygnp = np.meshgrid(np.arange(W), np.arange(H))

Xg = tf.constant(Xgnp.tolist(), dtype=tf.float32)
Yg = tf.constant(Ygnp.tolist(), dtype=tf.float32)

Vx = 10.5*tf.ones(R.shape, dtype=tf.float32)
Vy = 10.5*tf.ones(R.shape, dtype=tf.float32)

Flowx = tf.Variable(tf.random_normal([H,W], stddev=0.1), name="Flowx")
Flowy = tf.Variable(tf.random_normal([H,W], stddev=0.1), name="Flowy")

out = ofsampler(inp, Vx, Vy, Xg, Yg)
im_nn_out = ofsampler(inp, Flowx, Flowy, Xg, Yg)
loss = tf.reduce_mean(tf.square((im_nn_out-out)))

optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

im_shifted = out.eval(feed_dict={inp: im_})
plt.imshow(im_)
plt.pause(0.5)
plt.imshow(im_shifted)
plt.pause(0.5)

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
