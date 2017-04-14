from InterpNet import InterpNet
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pipi.flow import flow2hsv
import os
from utils import read_img, readFlow

mdl_height = int(384/4)
mdl_width = int(512/4)
batch_size = 8
verbose = 16
disp = 16
save = 32
lr = 1e-6
scale = 1.0
data_dir = "./../Chairs4"

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
        miss_list.append(curr_file + "miss.ppm")
        # print(flow_list[-1:][0])


learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
model_path = './models-in/in_sup_miss'
net = InterpNet(mdl_height=mdl_height, mdl_width=mdl_width, init_var=0.03)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.total_loss)
init = tf.global_variables_initializer()
check_op = tf.add_check_numerics_ops()
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# sess.run(init)
saver.restore(sess, model_path)

test_dir = './middlebury_data/Grove2'
img1 = read_img(test_dir + '/' + 'frame10.png')[0:mdl_height, 0:mdl_width, :]
img2 = read_img(test_dir + '/' + 'frame11.png')[0:mdl_height, 0:mdl_width, :]
edge = np.expand_dims(read_img(test_dir + '/' + 'frame10_edge.png')[0:mdl_height, 0:mdl_width], 2)
miss = np.expand_dims(np.mean(np.abs(img1-img2), axis=2), 2)

flow_gt = readFlow(test_dir + '/' + 'flow10.flo')[0:mdl_height, 0:mdl_width, :]
print('gt flow shape', flow_gt.shape)

flow = np.squeeze(net.outs[str(10)].eval(
            feed_dict={net.img1: np.expand_dims(img1, 0),
                       net.img2: np.expand_dims(img2, 0),
                       net.edge: np.expand_dims(edge, 0),
                       net.miss: np.expand_dims(miss, 0)}))

print('max computed flo is', np.max(flow))
print('does flow has nan?', np.any(np.isnan(flow)))
plt.imshow(np.hstack((img1, np.repeat(edge, 3, axis=2))))
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(1)
plt.imshow(np.hstack((flow2hsv(flow), flow2hsv(flow_gt))))
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.pause(1)
plt.close()

