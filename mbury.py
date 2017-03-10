import tensorflow as tf
import numpy as np
import skimage.io as io
import tflearn
from matplotlib import pyplot as plt
from tfsampler2 import ofsampler, batch_ofsampler

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

im = io.imread('eval-data/Dumptruck/frame10.png')
im_ = (im.astype(np.float32)/255).tolist()

im1 = io.imread('eval-data/Dumptruck/frame11.png')
im1_ = (im1.astype(np.float32)/255).tolist()

R = im[:,:,1]

ITER = 100000
VERBOSE = 5
LR = 0.0001
DECAY = 1000
alpha = 1
beta = 1
gamma = 2

W = im.shape[1]
H = im.shape[0]

Xgnp,Ygnp = np.meshgrid(np.arange(W), np.arange(H))

Hkernel = [[0,0,0,0],[0,-1,1,0],[0,0,0,0]]
Vkernel = [[0,0,0],[0,-1,0],[0,1,0],[0,0,0]]

Hplus = tf.reshape(tf.constant(Hkernel,dtype=tf.float32),[3,4,1,1])
Hminus = -tf.reshape(tf.constant(Hkernel,dtype=tf.float32),[3,4,1,1])
Vplus = tf.reshape(tf.constant(Vkernel,dtype=tf.float32),[4,3,1,1])
Vminus = -tf.reshape(tf.constant(Vkernel,dtype=tf.float32),[4,3,1,1])

Xg = tf.constant(Xgnp.tolist(), dtype=tf.float32)
Yg = tf.constant(Ygnp.tolist(), dtype=tf.float32)

Vx = 2.0*tf.ones(R.shape, dtype=tf.float32)
Vy = -2.0*tf.ones(R.shape, dtype=tf.float32)

inp1 = tf.placeholder(dtype=tf.float32, shape=[H,W,3])
inp2 = tf.placeholder(dtype=tf.float32, shape=[H,W,3])

diff_im = inp2-inp1
passive = (1-tf.reduce_mean(tf.square(diff_im), axis=2))

inp_nn = tf.expand_dims(axis=0,input=tf.concat(2,[inp1,inp2,diff_im]))

conv1 = tflearn.conv_2d(inp_nn, 16, [4, 4], activation='relu', name='conv1')
conv2 = tflearn.conv_2d(conv1, 16, [4, 4], activation='relu', name='conv2')

Flowx = tf.squeeze(tflearn.conv_2d(conv2, 1, [4, 4], activation='linear', name='Flowx'))
Flowy = tf.squeeze(tflearn.conv_2d(conv2, 1, [4, 4], activation='linear', name='Flowy'))

im_nn_out = ofsampler(inp1, Flowx, Flowy, Xg, Yg)
mse = tf.reduce_mean(tf.square((im_nn_out-inp2)))
reg1 =	tf.reduce_mean(tf.square(conv2d(tf.reshape(Flowx,[1,H,W,1]),Hplus)) + \
		tf.square(conv2d(tf.reshape(Flowx,[1,H,W,1]),Hminus)) + \
		tf.square(conv2d(tf.reshape(Flowx,[1,H,W,1]),Vplus)) + \
		tf.square(conv2d(tf.reshape(Flowx,[1,H,W,1]),Vminus)) + \
		tf.square(conv2d(tf.reshape(Flowy,[1,H,W,1]),Hplus)) + \
		tf.square(conv2d(tf.reshape(Flowy,[1,H,W,1]),Hminus)) + \
		tf.square(conv2d(tf.reshape(Flowy,[1,H,W,1]),Vplus)) + \
		tf.square(conv2d(tf.reshape(Flowy,[1,H,W,1]),Vminus)))
reg2 = tf.reduce_mean(passive*(tf.square(Flowx)+tf.square(Flowy)))
loss = alpha*mse + beta*reg1 + gamma*reg2


optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

plt.imshow(im_)
plt.pause(1)
plt.imshow(im1_)
plt.pause(1)
plt.clf()

# for i in range(ITER):
# 	if i%VERBOSE == 0:
# 	    train_accuracy = loss.eval(feed_dict={inp1:im_, inp2:im1_})
# 	    out_image = im_nn_out.eval(feed_dict={inp1:im_, inp2:im1_})
# 	    out_image1 = np.squeeze(np.array(out_image))
# 	 	# outx = np.squeeze(np.array(Flowx.eval(feed_dict={inp1:im_, inp2:im1_})))
# 		# outy = np.squeeze(np.array(Flowy.eval(feed_dict={inp1:im_, inp2:im1_})))
# 		plt.imshow(out_image1)
# 	    # plt.quiver(Xg,Yg,outx,outy)
# 	    plt.pause(0.0001)
# 	    print("step %d, training accuracy %g"%(i, train_accuracy))
# 	optimizer.run(feed_dict={inp1:im_, inp2:im1_})
# 	if i % DECAY == 0 and i is not 0:
# 	LR = 0.0001


for i in range(ITER):
	if i%VERBOSE == 0:
		train_accuracy = loss.eval(feed_dict={inp1:im_, inp2:im1_})
		out_image = im_nn_out.eval(feed_dict={inp1:im_, inp2:im1_})
		out_image1 = np.squeeze(np.array(out_image))
		plt.clf()
		
		if i%2==0:
			plt.imshow(passive.eval(feed_dict={inp1:im_, inp2:im1_}),cmap='gray')

		outx = -np.squeeze(np.array(Flowx.eval(feed_dict={inp1:im_, inp2:im1_})))
		outy = -np.squeeze(np.array(Flowy.eval(feed_dict={inp1:im_, inp2:im1_})))
		
		if i%2==1:
			plt.quiver(Xgnp[::20, ::20],Ygnp[::20, ::20],outx[::20, ::20],outy[::20, ::20],pivot='mid', units='inches')
		
		plt.pause(0.0001)
		print("step %d, training accuracy %g"%(i, train_accuracy))
	optimizer.run(feed_dict={inp1:im_, inp2:im1_})
	if i % DECAY == 0 and i is not 0:
		LR = LR