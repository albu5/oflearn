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
LR = 0.001
DECAY = 1000

W = im.shape[1]
H = im.shape[0]

Xgnp,Ygnp = np.meshgrid(np.arange(W), np.arange(H))

Xg = tf.constant(Xgnp.tolist(), dtype=tf.float32)
Yg = tf.constant(Ygnp.tolist(), dtype=tf.float32)

Vx = 5.0*tf.ones(R.shape, dtype=tf.float32)
Vy = 5.0*tf.ones(R.shape, dtype=tf.float32)

inp1 = tf.placeholder(dtype=tf.float32, shape=[H,W,3])
inp2 = tf.placeholder(dtype=tf.float32, shape=[H,W,3])

shift_op = (ofsampler(inp1, Vx, Vy, Xg, Yg))

inp_nn = tf.expand_dims(axis=0,input=tf.concat(2,[inp1,inp2]))

conv1 = tflearn.conv_2d(inp_nn, 16, [4, 4], activation='relu', name='conv1')
conv2 = tflearn.conv_2d(conv1, 16, [4, 4], activation='relu', name='conv2')

Flowx = tf.squeeze(tflearn.conv_2d(conv2, 1, [4, 4], activation='linear', name='Flowx'))
Flowy = tf.squeeze(tflearn.conv_2d(conv2, 1, [4, 4], activation='linear', name='Flowy'))

im_nn_out = ofsampler(inp1, Flowx, Flowy, Xg, Yg)
loss = tf.reduce_mean(tf.square((im_nn_out-inp2)))

optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

im_shifted = shift_op.eval(feed_dict={inp1: im_})
plt.imshow(im_)
plt.pause(1)
plt.imshow(im_shifted)
plt.pause(1)
plt.clf()

# for i in range(ITER):
# 	if i%VERBOSE == 0:
# 	    train_accuracy = loss.eval(feed_dict={inp1:im_, inp2:im_shifted})
# 	    out_image = im_nn_out.eval(feed_dict={inp1:im_, inp2:im_shifted})
# 	    out_image1 = np.squeeze(np.array(out_image))
# 	 	# outx = np.squeeze(np.array(Flowx.eval(feed_dict={inp1:im_, inp2:im_shifted})))
# 		# outy = np.squeeze(np.array(Flowy.eval(feed_dict={inp1:im_, inp2:im_shifted})))
# 		plt.imshow(out_image1)
# 	    # plt.quiver(Xg,Yg,outx,outy)
# 	    plt.pause(0.0001)
# 	    print("step %d, training accuracy %g"%(i, train_accuracy))
# 	optimizer.run(feed_dict={inp1:im_, inp2:im_shifted})
# 	if i % DECAY == 0 and i is not 0:
# 	LR = 0.0001


for i in range(ITER):
	if i%VERBOSE == 0:
		train_accuracy = loss.eval(feed_dict={inp1:im_, inp2:im_shifted})
		# out_image = im_nn_out.eval(feed_dict={inp1:im_, inp2:im_shifted})
		# out_image1 = np.squeeze(np.array(out_image))
		# plt.imshow(out_image1)

		outx = np.squeeze(np.array(Flowx.eval(feed_dict={inp1:im_, inp2:im_shifted})))
		outy = np.squeeze(np.array(Flowy.eval(feed_dict={inp1:im_, inp2:im_shifted})))
		plt.clf()
		plt.quiver(Xgnp[::20, ::20],Ygnp[::20, ::20],outx[::20, ::20],outy[::20, ::20],pivot='mid', units='inches')
		plt.pause(0.0001)
		print("step %d, training accuracy %g"%(i, train_accuracy))
	optimizer.run(feed_dict={inp1:im_, inp2:im_shifted})
	if i % DECAY == 0 and i is not 0:
		LR = 0.0001