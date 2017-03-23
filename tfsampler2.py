import tensorflow as tf


def get_values_at_coordinates(input, coordinates):
	input_as_vector = tf.reshape(input, [-1])
	coordinates_as_indices = (coordinates[:, 0] * tf.shape(input)[1]) + coordinates[:, 1]
	return tf.gather(input_as_vector, coordinates_as_indices)

def imsampler_gray(I, C):
	top_left = tf.cast(tf.floor(C), tf.int32)

	top_right = tf.cast(
	    tf.concat(axis=1, values=[tf.floor(C[:, 0:1]), tf.ceil(C[:, 1:2])]), tf.int32)

	bottom_left = tf.cast(
	    tf.concat(axis=1, values=[tf.ceil(C[:, 0:1]), tf.floor(C[:, 1:2])]), tf.int32)

	bottom_right = tf.cast(tf.ceil(C), tf.int32)

	values_at_top_left = get_values_at_coordinates(I, top_left)
	values_at_top_right = get_values_at_coordinates(I, top_right)
	values_at_bottom_left = get_values_at_coordinates(I, bottom_left)
	values_at_bottom_right = get_values_at_coordinates(I, bottom_right)

	# Varies between 0.0 and 1.0.
	horizontal_offset = C[:, 0] - tf.cast(top_left[:, 0], tf.float32)

	horizontal_interpolated_top = (
	    ((1.0 - horizontal_offset) * values_at_top_left)
	    + (horizontal_offset * values_at_top_right))

	horizontal_interpolated_bottom = (
	    ((1.0 - horizontal_offset) * values_at_bottom_left)
	    + (horizontal_offset * values_at_bottom_right))

	vertical_offset = C[:, 1] - tf.cast(top_left[:, 1], tf.float32)

	interpolated_result = (
	    ((1.0 - vertical_offset) * horizontal_interpolated_top)
	    + (vertical_offset * horizontal_interpolated_bottom))
	return interpolated_result

def imsampler(I,C):
	R,G,B = tf.unstack(I,axis=2)
	
	Ri = tf.reshape(imsampler_gray(R, C), tf.shape(R))
	Gi = tf.reshape(imsampler_gray(G, C), tf.shape(R))
	Bi = tf.reshape(imsampler_gray(B, C), tf.shape(R))

	return tf.stack([Ri,Gi,Bi], axis=2)

def ofsampler(im, Vx, Vy, Xg, Yg):
	Xgrid = tf.clip_by_value(Xg+Vx, 0.0, tf.cast((tf.shape(im)[1]-1), dtype=tf.float32) )
	Ygrid = tf.clip_by_value(Yg+Vy, 0.0, tf.cast((tf.shape(im)[0]-1), dtype=tf.float32) )
	C = tf.stack( [tf.reshape(Ygrid, [-1]), tf.reshape(Xgrid, [-1])], axis=1)
	return imsampler(im, C)

def batch_ofsampler(im_bacth, Vx_batch, Vy_batch, Xg_batch, Yg_batch):
	elems = (im_bacth, Vx_batch, Vy_batch, Xg_batch, Yg_batch)
	return(tf.map_fn(lambda x: ofsampler(x[0],x[1],x[2],x[3],x[4]), elems=elems, dtype=tf.float32))