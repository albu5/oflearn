from FlowNet import FlowNetS
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pipi.flow import readFlow, flow2hsv
import os
from utils import read_img
from scipy.misc import imresize

WIDTH = 512
HEIGHT = 384
BS = 16
LR = 1e-6
VERBOSE = 16
DISP = 16
FLO_MAX = 1e8
SAVE = 32


learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
model_path = './models/fn_sup'
net = FlowNetS(HEIGHT, WIDTH)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
# sess.run(init)

# saver = tf.train.import_meta_graph(model_path + '.meta')
saver.restore(sess, model_path)


test_dir = './middlebury_data/Grove3'
img1 = read_img(test_dir + '/' + 'frame10.png')[0:HEIGHT, 0:WIDTH, :]
img2 = read_img(test_dir + '/' + 'frame11.png')[0:HEIGHT, 0:WIDTH, :]
flow_gt = readFlow(test_dir + '/' + 'flow10.flo')[0:HEIGHT, 0:WIDTH, :]
print('gt flow shape', flow_gt.shape)

flow_eval = np.squeeze(net.flow2.eval(feed_dict={net.inp1: np.expand_dims(img1, 0),
                                                 net.inp2: np.expand_dims(img2, 0)}))
print('eval flow shape', flow_eval.shape)
img_vis = np.hstack((img1, img2))
flow_vis = np.hstack((imresize(flow2hsv(flow_gt), 1./4), flow2hsv(-flow_eval)))
plt.imshow(img_vis)
plt.pause(2)
plt.imshow(flow_vis)
plt.pause(2)


