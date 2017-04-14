from FlowNetEdge import FlowNetS
import tensorflow as tf
import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
from pipi.flow import readFlow, flow2hsv
import os
from random import randint


def get_batch(BS, img1_list, img2_list, flow_list, HEIGHT, WIDTH, FLO_MAX):
    N = len(img2_list)
    I1 = np.zeros((BS, HEIGHT, WIDTH, 3), dtype=np.float32)
    I2 = np.zeros((BS, HEIGHT, WIDTH, 3), dtype=np.float32)
    F = np.zeros((BS, HEIGHT, WIDTH, 2), dtype=np.float32)
    E = np.zeros((BS, HEIGHT, WIDTH, 1), dtype=np.float32)
    good_ids = []
    for i in range(BS):
        try:
            idx = randint(0, N - 1)
            img1 = io.imread(img1_list[idx]).astype(np.float32) / 255
            img2 = io.imread(img2_list[idx]).astype(np.float32) / 255
            edge = io.imread(img2_list[idx]).astype(np.float32) / 255
            flow = (readFlow(flow_list[idx]).astype(np.float32))
            I1[i][:][:][:] = img1
            I2[i][:][:][:] = img2
            F[i][:][:][:] = flow
            E[i][:][:][0] = edge
            good_ids.append(True)
        except:
            print('Some images were not read')
            good_ids.append(False)
    I1 = I1[good_ids, :, :, :]
    I2 = I2[good_ids, :, :, :]
    F = F[good_ids, :, :, :]
    E = E[good_ids, :, :, :]
    return I1, I2, F, E


DATA_DIR = "FlyingChairs_release/data/"
WIDTH = 512
HEIGHT = 384
BS = 16
LR = 1e-6
VERBOSE = 16
DISP = 16
FLO_MAX = 1e8
SAVE = 32

img1_list = []
img2_list = []
flow_list = []
edge_list = []

for file in os.listdir(DATA_DIR):
    if file.endswith(".flo"):
        curr_file = os.path.join(DATA_DIR, file)
        curr_file = curr_file[:-8]
        flow_list.append(curr_file + "flow.flo")
        img1_list.append(curr_file + "img1.ppm")
        img2_list.append(curr_file + "img2.ppm")
        edge_list.append(curr_file + 'edge.ppm')
    # print(flow_list[-1:][0])

learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
model_path = './models-fn//fn_edge_sup'
net = FlowNetS(HEIGHT, WIDTH)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(net.loss)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(init)

# saver = tf.train.import_meta_graph(model_path + '.meta')
# saver.restore(sess, model_path)

niter = 0

# I1,I2,F = get_batch(BS, img1_list, img2_list, flow_list, HEIGHT, WIDTH)
# print(I1.shape,F.shape)
while True:
    I1, I2, F, E = get_batch(BS, img1_list, img2_list, flow_list, HEIGHT, WIDTH, FLO_MAX)
    # print('max gt flow is', np.max(F))

    if niter % DISP == 0:
        flow = np.squeeze(net.flow2.eval(
            feed_dict={net.inp1: np.expand_dims(I1[0][:][:][:], 0),
                       net.inp2: np.expand_dims(I2[0][:][:][:], 0),
                       net.edge: np.expand_dims(E[0][:][:][:], 0),
                       }))

        print('max computed flo is', np.max(flow))
        print('does flow has nan?', np.any(np.isnan(flow)))
        plt.imshow(I1[0][:][:][:])
        plt.pause(1)
        plt.imshow(flow2hsv(flow))
        plt.pause(1)
        plt.close()

    if niter % VERBOSE == 0:
        flow_loss = np.squeeze(net.loss.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F, net.edge: E}))
        # flow_loss2 = np.squeeze(net.loss2.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss3 = np.squeeze(net.loss3.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss4 = np.squeeze(net.loss4.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss5 = np.squeeze(net.loss5.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))

        print('losses are ', flow_loss)
    # print('losses are ', flow_loss, flow_loss2, flow_loss3, flow_loss4, flow_loss5)

    if niter % SAVE == 0:
        saver.save(sess, model_path)

    sess.run(optimizer, feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F, learning_rate: LR, net.edge: E})
    niter += 1
    print(BS * niter, 'image pairs trained')
