from FlowNet import FlowNetS
import tensorflow as tf
import numpy as np
import skimage.io as io
# import tflearn
from matplotlib import pyplot as plt
from tfsampler2 import ofsampler, batch_ofsampler
from pipi.flow import flow2hsv

im = io.imread('FlyingChairs_release/data/13871_img1.ppm')
im_ = (im.astype(np.float32)/255)
im_ = im_[0:383][0:511][:]
print(im_.shape)
# im__ = [im_.tolist()]
im__ = np.expand_dims(im_,0)

im2 = io.imread('FlyingChairs_release/data/13871_img2.ppm')
im2_ = (im2.astype(np.float32)/255)
im2_ = im2_[0:383][0:511][:]
print(im2_.shape)
# im__ = [im_.tolist()]
im2__ = np.expand_dims(im2_,0)

net = FlowNetS(im_.shape[0], im_.shape[1])


init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

flow = np.squeeze(net.flow2.eval(feed_dict={net.inp1:im__, net.inp2:im2__}))
print(flow.shape)
plt.imshow(im_)
plt.pause(1)
plt.imshow(flow2hsv(flow))
plt.pause(1)
print(np.any(np.isnan(flow)))