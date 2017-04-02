import tensorflow as tf
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

boundaries_np = np.zeros(shape=(1, 500, 250, 1), dtype=np.float32)
boundaries_tf = tf.constant(boundaries_np, dtype=tf.float32)

rshift2 = [[0., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 0.]]
dshift2 = [[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 1.],
           [0., 0., 0.]]

rshift2 = [[0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0]]
dshift2 = [[0, 0],
           [0, 0],
           [0, 0],
           [1, 0]]


rshift2 = np.array(rshift2, dtype=np.float32)
dshift2 = np.array(dshift2, dtype=np.float32)

rshift4 = np.expand_dims(rshift2, axis=2)
rshift4 = np.expand_dims(rshift4, axis=3)
rshift4 = np.concatenate((np.concatenate((rshift4*1, rshift4*0, rshift4*0), axis=2),
                          np.concatenate((rshift4*0, rshift4*1, rshift4*0), axis=2),
                          np.concatenate((rshift4*0, rshift4*0, rshift4*1), axis=2)), axis=3)

dshift4 = np.expand_dims(dshift2, axis=2)
dshift4 = np.expand_dims(dshift4, axis=3)
dshift4 = np.concatenate((np.concatenate((dshift4*1, dshift4*0, dshift4*0), axis=2),
                          np.concatenate((dshift4*0, dshift4*1, dshift4*0), axis=2),
                          np.concatenate((dshift4*0, dshift4*0, dshift4*1), axis=2)), axis=3)

print(rshift4.shape)
print(dshift4.shape)

rshift4_tf = tf.constant(rshift4, dtype=tf.float32)
dshift4_tf = tf.constant(dshift4, dtype=tf.float32)

img1 = np.expand_dims(io.imread('./sample_im/Lenna.png'), axis=0).astype(np.float32) / 255
W = img1.shape[2]
H = img1.shape[1]
C = img1.shape[3]

inp = tf.placeholder(dtype=tf.float32, shape=[None, H, W, C])
out_r = tf.nn.conv2d(inp, filter=rshift4_tf, padding='SAME', strides=[1, 1, 1, 1])
out_d = tf.nn.conv2d(inp, filter=dshift4_tf, padding='SAME', strides=[1, 1, 1, 1])

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
img1_r = out_r.eval(feed_dict={inp: img1})
img1_d = out_d.eval(feed_dict={inp: img1})

I = img1[0][:][:][:]
R = img1_r[0][:][:][:]
D = img1_d[0][:][:][:]

plt.imshow(I)
plt.pause(interval=5)
plt.imshow(R)
plt.pause(interval=5)
plt.imshow(D)
plt.pause(interval=5)

print(np.max(I), np.min(I))
print(np.max(R), np.min(R))
print(np.max(D), np.min(D))

