from InterpNet import InterpNet
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from pipi.flow import flow2hsv
import os
from utils import get_batch

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
model_path = './models-in/in_sup_miss'


niter = 0

I1, I2, E, M, F = get_batch(batch_size,
                            img1_list, img2_list, edge_list, miss_list, flow_list,
                            height=mdl_height, width=mdl_width)
print(I1.shape, I2.shape, E.shape, M.shape, F.shape)

batch_acc_t = []

while True:
    I1, I2, E, M, F = get_batch(batch_size,
                                img1_list, img2_list, edge_list, miss_list, flow_list,
                                height=mdl_height, width=mdl_width)

    # print('max gt flow is', np.max(F))

    if niter % disp == 0:
        flow = np.squeeze(net.outs[str(10)].eval(
            feed_dict={net.img1: np.expand_dims(I1[0][:][:][:], 0),
                       net.img2: np.expand_dims(I2[0][:][:][:], 0),
                       net.miss: np.expand_dims(M[0][:][:][:], 0),
                       net.edge: np.expand_dims(E[0][:][:][:], 0)}))

        print('max computed flo is', np.max(flow))
        print('does flow has nan?', np.any(np.isnan(flow)))
        plt.imshow(np.hstack((I1[0][:][:][:], np.repeat(E[0][:][:][:], 3, axis=2))))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(1)
        plt.imshow(np.hstack((flow2hsv(F[0][:][:][:]),
                              flow2hsv(flow))))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(1)
        plt.close()

    if niter % verbose == 0:
        flow_loss = np.squeeze(net.total_loss.eval(feed_dict={net.img1: I1,
                                                              net.img2: I2,
                                                              net.edge: E,
                                                              net.miss: M,
                                                              net.flow: F}))
        batch_acc_t.append(flow_loss)
        plt.plot(batch_acc_t)
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.pause(1)
        plt.close()
        # plt.close()
        # flow_loss2 = np.squeeze(net.loss2.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss3 = np.squeeze(net.loss3.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss4 = np.squeeze(net.loss4.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))
        # flow_loss5 = np.squeeze(net.loss5.eval(feed_dict={net.inp1: I1, net.inp2: I2, net.gt: F}))

        # print('losses are ', flow_loss)
        # print('losses are ', flow_loss, flow_loss2, flow_loss3, flow_loss4, flow_loss5)

    if niter % save == 0:
        saver.save(sess, model_path)
    try:
        sess.run([optimizer], feed_dict={net.img1: I1,
                                         net.img2: I2,
                                         net.edge: E,
                                         net.miss: M,
                                         net.flow: F,
                                         learning_rate: lr})
    except tf.errors.InvalidArgumentError as e:
        print(e.op.inputs[0])
        break
    niter += 1
    print(batch_size * niter, 'image pairs trained')
