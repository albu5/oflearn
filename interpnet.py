import tensorflow as tf
import numpy as np
#
# mdl_height = 384/4
# mdl_width = 512/4
# batch_size = 16
# verbose = 16
# disp = 16
# save = 32

eps = 1e-8


def rshift_flow_kernel():
    rshift2 = [[0, 1],
               [0, 0]]
    rshift2 = np.array(rshift2, dtype=np.float32)
    rshift4 = np.expand_dims(rshift2, axis=2)
    rshift4 = np.expand_dims(rshift4, axis=3)
    rshift4 = np.concatenate((np.concatenate((rshift4 * 1, rshift4 * 0), axis=2),
                              np.concatenate((rshift4 * 0, rshift4 * 1), axis=2)), axis=3)
    return tf.constant(rshift4, dtype=tf.float32)


def dshift_flow_kernel():
    rshift2 = [[0, 0],
               [1, 0]]
    rshift2 = np.array(rshift2, dtype=np.float32)
    rshift4 = np.expand_dims(rshift2, axis=2)
    rshift4 = np.expand_dims(rshift4, axis=3)
    rshift4 = np.concatenate((np.concatenate((rshift4 * 1, rshift4 * 0), axis=2),
                              np.concatenate((rshift4 * 0, rshift4 * 1), axis=2)), axis=3)
    return tf.constant(rshift4, dtype=tf.float32)


def flow_boundary(mdl_height, mdl_width):
    boundaries = np.ones(shape=(1, mdl_height, mdl_width), dtype=np.float32)
    print(boundaries.shape)
    boundaries[0, 0, :] = 0
    boundaries[0, mdl_height - 1, :] = 0
    boundaries[0, :, mdl_width - 1] = 0
    boundaries[0, :, 0] = 0
    return tf.constant(boundaries, dtype=tf.float32)


def conv2d(x, w, b, strides=1, use_bias=True):
    # Conv2D wrapper, with bias
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    if use_bias:
        x = tf.nn.bias_add(x, b)
    return x


def epe(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=3) + eps)


def epe_loss(x, y):
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=3)))


# complete this
'''
def ldl_loss(x, y):
    dims = tf.shape(x)
    x_slice_horz = tf.slice(x, [0, 0, 0, 0], [dims[0], dims[1], dims[2] - 1, dims[3]])
    x_slice_left = tf.slice(x, [0, 0, 1, 0], [dims[0], dims[1], dims[2] - 1, dims[3]])

    x_slice_vert = tf.slice(x, [0, 0, 0, 0], [dims[0], dims[1] - 1, dims[2], dims[3]])
    x_slice_down = tf.slice(x, [0, 1, 0, 0], [dims[0], dims[1] - 1, dims[2], dims[3]])

    y_slice_horz = tf.slice(y, [0, 0, 0, 0], [dims[0], dims[1], dims[2] - 1, dims[3]])
    y_slice_left = tf.slice(y, [0, 0, 1, 0], [dims[0], dims[1], dims[2] - 1, dims[3]])

    y_slice_vert = tf.slice(y, [0, 0, 0, 0], [dims[0], dims[1] - 1, dims[2], dims[3]])
    y_slice_down = tf.slice(y, [0, 1, 0, 0], [dims[0], dims[1] - 1, dims[2], dims[3]])

    loss_matrix = tf.abs(epe_loss(x_slice_horz, x_slice_left) - epe_loss(y_slice_horz, y_slice_left)) + \
        tf.abs(epe_loss(x_slice_vert, x_slice_down) - epe_loss(y_slice_vert, y_slice_down))
    return tf.reduce_mean(loss_matrix)
'''


def ldl_loss_dbg(x, y, h_kernel, v_kernel):
    x_r = tf.nn.conv2d(x, filter=h_kernel, padding='SAME', strides=[1, 1, 1, 1], name='x_r')
    # x_d = tf.nn.conv2d(x, filter=v_kernel, padding='SAME', strides=[1, 1, 1, 1], name='x_d')
    # y_r = tf.nn.conv2d(y, filter=h_kernel, padding='SAME', strides=[1, 1, 1, 1], name='y_r')
    # y_d = tf.nn.conv2d(y, filter=v_kernel, padding='SAME', strides=[1, 1, 1, 1], name='y_d')
    return x_r


def ldl_loss(x, y, h_kernel, v_kernel, boundary):
    x_r = tf.nn.conv2d(x, filter=h_kernel, padding='SAME', strides=[1, 1, 1, 1], name='x_r')
    x_d = tf.nn.conv2d(x, filter=v_kernel, padding='SAME', strides=[1, 1, 1, 1], name='x_d')
    y_r = tf.nn.conv2d(y, filter=h_kernel, padding='SAME', strides=[1, 1, 1, 1], name='y_r')
    y_d = tf.nn.conv2d(y, filter=v_kernel, padding='SAME', strides=[1, 1, 1, 1], name='y_d')

    # print_ops = [tf.Print(x_r, ['inf x_r', tf.reduce_any(tf.is_inf(x_r)), tf.reduce_max(x_r)]),
    #              tf.Print(x_r, ['nan x_r', tf.reduce_any(tf.is_nan(x_r)), tf.reduce_max(x_r)])]
    # tf.Print(x_d, [tf.reduce_any(tf.is_inf(x_d))])
    # tf.Print(x_d, tf.reduce_any(tf.is_nan(x_d)))
    # tf.Print(y_r, tf.reduce_any(tf.is_inf(y_r)))
    # tf.Print(y_r, tf.reduce_any(tf.is_nan(y_r)))
    # tf.Print(y_d, tf.reduce_any(tf.is_inf(y_d)))
    # tf.Print(y_d, tf.reduce_any(tf.is_nan(y_d)))

    loss_matrix = tf.square(epe(x, x_r) - epe(y, y_r)) + tf.square(epe(x, x_d) - epe(y, y_d))

    batch_size = tf.shape(x)
    boundary = tf.tile(boundary, [batch_size[0], 1, 1])

    loss_matrix = tf.multiply(boundary, loss_matrix)

    return tf.reduce_mean(loss_matrix)


class InterpNet:
    def __init__(self, mdl_width, mdl_height, init_var):
        self.flow_h_shift_kernel = rshift_flow_kernel()
        self.flow_v_shift_kernel = rshift_flow_kernel()
        self.flow_boundary = flow_boundary(mdl_height, mdl_width)

        self.img1 = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, 3])
        self.img2 = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, 3])
        self.flow = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, 2])
        self.edge = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, 1])
        self.miss = tf.placeholder(dtype=tf.float32, shape=[None, mdl_height, mdl_width, 1])

        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.outs = {}
        self.epe = {}
        self.ldl = {}
        self.total_loss = 0

        n_channels = [8, 32, 64, 128, 128, 128, 128, 256, 256, 256, 256, 2]

        self.layers[str(0)] = tf.concat(values=[self.img1, self.img2, self.edge, self.miss], axis=3)

        for i in range(10):
            self.weights[str(i+1)] = tf.Variable(tf.truncated_normal(stddev=init_var,
                                                                     shape=[7, 7, n_channels[i], n_channels[i+1]]))
            self.biases[str(i+1)] = tf.Variable(tf.truncated_normal(stddev=init_var, shape=[n_channels[i+1]]))

            self.weights['out' + str(i+1)] = tf.Variable(tf.truncated_normal(stddev=init_var,
                                                                             shape=[7, 7, n_channels[i+1], 2]))
            self.biases['out' + str(i+1)] = tf.Variable(tf.truncated_normal(stddev=init_var, shape=[2]))

            self.layers[str(i+1)] = tf.nn.elu(conv2d(self.layers[str(i)],
                                                     self.weights[str(i+1)],
                                                     self.biases[str(i+1)]))
            self.outs[str(i+1)] = conv2d(self.layers[str(i+1)],
                                         self.weights['out' + str(i+1)],
                                         self.biases['out' + str(i+1)])
            self.epe[str(i+1)] = epe_loss(self.outs[str(i+1)], self.flow)
            self.ldl[str(i+1)] = ldl_loss(self.outs[str(i+1)], self.flow,
                                          self.flow_h_shift_kernel, self.flow_v_shift_kernel, self.flow_boundary)
            self.total_loss += self.epe[str(i+1)] + self.ldl[str(i+1)]
        self.total_loss += self.epe[str(10)] + self.ldl[str(10)]
        # self.dbg = ldl_loss_dbg(self.outs[str(10)], self.flow,
        #                         self.flow_h_shift_kernel, self.flow_v_shift_kernel)


