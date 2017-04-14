from utils import get_batch_jitter_scale
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from pipi.flow import flow2hsv
from scipy.io import savemat


mdl_height = int(384)
mdl_width = int(512)
batch_size = 1
verbose = 16
disp = 16
save = 32
lr = 1e-5
scale = 1./4
data_dir = "./FlyingChairs_release/data/"

img1_list = []
img2_list = []
flow_list = []
edge_list = []
miss_list = []

for file in os.listdir(data_dir):
    if file.endswith(".flo"):
        curr_file = os.path.join(data_dir, file)
        curr_file = curr_file[:-8]
        flow_list.append(curr_file + "flow.flo")
        img1_list.append(curr_file + "img1.ppm")
        img2_list.append(curr_file + "img2.ppm")
        edge_list.append(curr_file + "edge.ppm")
        # print(flow_list[-1:][0])

I1, I2, E, F = get_batch_jitter_scale(1,
                                      img1_list, img2_list, edge_list, flow_list,
                                      height=mdl_height, width=mdl_width)
F = F * scale

I = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, None])
O = tf.image.resize_bilinear(I, size=[int(mdl_height * scale), int(mdl_width * scale)])
sess = tf.InteractiveSession()
I1_ = np.squeeze(O.eval(feed_dict={I: I1}))
I2_ = np.squeeze(O.eval(feed_dict={I: I2}))
E_ = np.squeeze(O.eval(feed_dict={I: E}))
F_ = np.squeeze(O.eval(feed_dict={I: F}))

print(I1_.shape, I2_.shape, E_.shape, F_.shape)
plt.imshow(np.squeeze(I1_))
plt.pause(1)
plt.imshow(np.squeeze(I2_))
plt.pause(1)
plt.imshow(np.squeeze(E_))
plt.pause(1)
plt.imshow(flow2hsv(np.squeeze(F_)))
plt.pause(1)

savemat('./tmp/batch_data.mat', mdict={'img1': I1_,
                                       'img2': I2_,
                                       'edge': E_,
                                       'flow': F_})
