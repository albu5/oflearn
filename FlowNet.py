import tensorflow as tf

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def flow_loss(flow, gt, shape):
	gt_down = tf.image.resize_bilinear(gt, shape)
	return tf.reduce_sum(tf.square(flow-gt_down))

class FlowNetS:
	def __init__(self, H ,W):
		# input layers
		self.inp1 = tf.placeholder(dtype=tf.float32, shape=[None, H, W, 3])
		self.inp2 = tf.placeholder(dtype=tf.float32, shape=[None, H, W, 3])
		self.inp2 = tf.placeholder(dtype=tf.float32, shape=[None, H, W, 2])

		# concat the inputs 384->384
		self.concat_inp = tf.concat(values=[self.inp1, self.inp2], axis=3)

		# conv1 384->192
		self.W_conv_1 = tf.Variable(tf.truncated_normal([7, 7, 6, 64]), name='W_conv_1')
		self.B_conv_1 = tf.Variable(tf.truncated_normal([64]), name='B_conv_1')
		self.conv1 = conv2d(self.concat_inp, self.W_conv_1, self.B_conv_1, 1)
		self.pool1 = maxpool2d(self.conv1, 2)

		# conv2 192->96
		self.W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 64, 128]), name='W_conv_2')
		self.B_conv_2 = tf.Variable(tf.truncated_normal([128]), name='B_conv_2')
		self.conv2 = conv2d(self.pool1, self.W_conv_2, self.B_conv_2, 1)
		self.pool2 = maxpool2d(self.conv2, 2)

		# conv2_1 96->48
		self.W_conv_2_1 = tf.Variable(tf.truncated_normal([5, 5, 128, 256]), name='W_conv_2_1')
		self.B_conv_2_1 = tf.Variable(tf.truncated_normal([256]), name='B_conv_2_1')
		self.conv2_1 = conv2d(self.pool2, self.W_conv_2_1, self.B_conv_2_1, 1)
		self.pool2_1 = maxpool2d(self.conv2_1, 2)

		# conv3 48->48
		self.W_conv_3 = tf.Variable(tf.truncated_normal([3, 3, 256, 256]), name='W_conv_3')
		self.B_conv_3 = tf.Variable(tf.truncated_normal([256]), name='B_conv_3')
		self.conv3 = conv2d(self.pool2_1, self.W_conv_3, self.B_conv_3, 1)
		self.pool3 = maxpool2d(self.conv3, 1)

		# conv3_1 48->24
		self.W_conv_3_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 512]), name='W_conv_3_1')
		self.B_conv_3_1 = tf.Variable(tf.truncated_normal([512]), name='B_conv_3_1')
		self.conv3_1 = conv2d(self.pool3, self.W_conv_3_1, self.B_conv_3_1, 1)
		self.pool3_1 = maxpool2d(self.conv3_1, 2)

		# conv4 24->24
		self.W_conv_4 = tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='W_conv_4')
		self.B_conv_4 = tf.Variable(tf.truncated_normal([512]), name='B_conv_4')
		self.conv4 = conv2d(self.pool3_1, self.W_conv_4, self.B_conv_4, 1)
		self.pool4 = maxpool2d(self.conv4, 1)

		# conv4_1 24->12
		self.W_conv_4_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='W_conv_4_1')
		self.B_conv_4_1 = tf.Variable(tf.truncated_normal([512]), name='B_conv_4_1')
		self.conv4_1 = conv2d(self.pool4, self.W_conv_4_1, self.B_conv_4_1, 1)
		self.pool4_1 = maxpool2d(self.conv4_1, 2)

		# conv5 12->12
		self.W_conv_5 = tf.Variable(tf.truncated_normal([3, 3, 512, 512]), name='W_conv_5')
		self.B_conv_5 = tf.Variable(tf.truncated_normal([512]), name='B_conv_5')
		self.conv5 = conv2d(self.pool4_1, self.W_conv_5, self.B_conv_5, 1)
		self.pool5 = maxpool2d(self.conv5, 1)

		# conv5_1 12->6
		self.W_conv_5_1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1024]), name='W_conv_5_1')
		self.B_conv_5_1 = tf.Variable(tf.truncated_normal([1024]), name='B_conv_5_1')
		self.conv5_1 = conv2d(self.pool5, self.W_conv_5_1, self.B_conv_5_1, 1)
		self.pool5_1 = maxpool2d(self.conv5_1, 1)

		self.W_deconv_5 = tf.Variable(tf.truncated_normal([5, 5, 512, 1024]), name='W_deconv_5')
		self.deconv5 = tf.nn.conv2d_transpose(self.pool5_1, self.W_deconv_5, [-1, 12, 16, 512], strides=[1, 1, 1, 1], padding='SAME')

		# -1x12x16x1024
		self.cat5 = tf.concat(values=[self.deconv5, self.pool5], axis=3)
		
		# -1x12x16x2
		self.W_pred5 = tf.Variable(tf.truncated_normal([5, 5, 512, 2]), name='W_pred5')
		self.B_pred5 = tf.Variable(tf.truncated_normal([2]), name='B_pred5')
		self.flow5 = tf.nn.bias_add(tf.nn.conv2d(self.cat5, self.W_pred5, [1,1,1,1], padding='SAME'), self.B_pred5)
		
		self.flow5x2 = tf.image.resize_bilinear(self.flow5, [-1, 24, 32, 2])

		# -1x24x32x256
		self.W_deconv_4 = tf.Variable(tf.truncated_normal([5, 5, 256, 512]), name='W_deconv_4')
		self.cat5_up = tf.nn.conv2d_transpose(self.cat5, self.W_deconv_4, [-1, 24, 32, 256], strides=[1, 1, 1, 1], padding='SAME')

		# -1x24x32x770(256+512+2)
		self.cat4 = tf.concat(values=[self.cat5_up, self.pool4, self.flow5x2], axis=3)

		# -1x24x32x2
		self.W_pred4 = tf.Variable(tf.truncated_normal([5, 5, 770, 2]), name='W_pred4')
		self.B_pred4 = tf.Variable(tf.truncated_normal([2]), name='B_pred4')
		self.flow4 = tf.nn.bias_add(tf.nn.conv2d(self.cat4, self.W_pred4, [1,1,1,1], padding='SAME'), self.B_pred4)

		self.flow4x2 = tf.image.resize_bilinear(self.flow4, [-1, 48, 64, 2])
		
		# -1x48x64x128
		self.W_deconv_3 = tf.Variable(tf.truncated_normal([5, 5, 128, 770]), name='W_deconv_3')
		self.cat4_up = tf.nn.conv2d_transpose(self.cat4, self.W_deconv_3, [-1, 48, 64, 128], strides=[1, 1, 1, 1], padding='SAME')

		# -1x48x64x386(128+256+2)
		self.cat3 = tf.concat(values=[self.cat4_up, self.pool3, self.flow4x2], axis=3)

		# -1x48x64x2
		self.W_pred3 = tf.Variable(tf.truncated_normal([5, 5, 386, 2]), name='W_pred3')
		self.B_pred3 = tf.Variable(tf.truncated_normal([2]), name='B_pred3')
		self.flow3 = tf.nn.bias_add(tf.nn.conv2d(self.cat3, self.W_pred3, [1,1,1,1], padding='SAME'), self.B_pred3)

		self.flow3x2 = tf.image.resize_bilinear(self.flow3, [-1, 96, 128, 2])

		# -1x96x128x128
		self.W_deconv_3 = tf.Variable(tf.truncated_normal([5, 5, 64, 386]), name='W_deconv_3')
		self.cat3_up = tf.nn.conv2d_transpose(self.cat3, self.W_deconv_3, [-1, 96, 128, 64], strides=[1, 1, 1, 1], padding='SAME')

		# -1x96x128x194(64+128+2)
		self.cat2 = tf.concat(values=[self.cat3_up, self.pool2, self.flow3x2], axis=3)

		# -1x96x128x2
		self.W_pred2 = tf.Variable(tf.truncated_normal([5, 5, 194, 2]), name='W_pred2')
		self.B_pred2 = tf.Variable(tf.truncated_normal([2]), name='B_pred2')
		self.flow2 = tf.nn.bias_add(tf.nn.conv2d(self.cat2, self.W_pred2, [1,1,1,1], padding='SAME'), self.B_pred2)

		self.loss2 = flow_loss(self.flow2, gt, [96, 128])
		self.loss3 = flow_loss(self.flow3, gt, [48, 64])
		self.loss4 = flow_loss(self.flow4, gt, [24, 32])
		self.loss5 = flow_loss(self.flow5, gt, [12, 16])

		weights = [0.01, 0.02, 0.08, 0.32];
		self.loss = weights[0]*self.loss5 + \
					weights[1]*self.loss4 + \
					weights[2]*self.loss3 + \
					weights[3]*self.loss2
					
