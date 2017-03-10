import tensorflow as tf
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from tfsampler2 import ofsampler, batch_ofsampler

im = io.imread('sample.png')
im_ = (im.astype(np.float32)/255).tolist()
R = im[:,:,1]

inp = tf.placeholder(dtype=tf.float32, shape=[R.shape[0], R.shape[1], 3])
W = im.shape[1]
H = im.shape[0]

Xgnp,Ygnp = np.meshgrid(np.arange(W), np.arange(H))

Xg = tf.constant(Xgnp.tolist(), dtype=tf.float32)
Yg = tf.constant(Ygnp.tolist(), dtype=tf.float32)

Vx = 10.5*tf.ones(R.shape, dtype=tf.float32)
Vy = 10.5*tf.ones(R.shape, dtype=tf.float32)

out = ofsampler(inp, Vx, Vy, Xg, Yg)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

im2 = out.eval(feed_dict={inp: im_})
plt.imshow(im2)
plt.pause(10)