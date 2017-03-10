import tensorflow as tf

def batch_ofsampler(im_bacth, Vx_batch, Vy_batch, Xg_batch, Yg_batch):
	elems = (im_bacth, Vx_batch, Vy_batch, Xg_batch, Yg_batch)
	return(tf.map_fn(lambda x: ofsampler(x[0],x[1],x[2],x[3],x[4]), elems=elems, dtype=tf.float32))

def ofsampler(im, Vx, Vy, Xg, Yg):
	Xgrid = tf.clip_by_value(Xg+Vx, 0, tf.cast((tf.shape(im)[1]-1), dtype=tf.float32))
	Ygrid = tf.clip_by_value(Yg+Vy, 0, tf.cast((tf.shape(im)[0]-1), dtype=tf.float32))
	return imsampler(im, Xgrid, Ygrid)

def imsampler(im, Xgrid, Ygrid):
	R,G,B = tf.unstack(im, axis=2)

	# get corner coordinates
	C1x = tf.floor(Xgrid)
	C1y = tf.floor(Ygrid)

	C2x = tf.ceil(Xgrid)
	C2y = tf.ceil(Ygrid)

	C3x = tf.ceil(Xgrid)
	C3y = tf.floor(Ygrid)

	C4x = tf.floor(Xgrid)
	C4y = tf.ceil(Ygrid)

	xoff = Xgrid-tf.cast(C2x, tf.float32)
	yoff = Ygrid-tf.cast(C2y, tf.float32)

	xoff = tf.Print(xoff, [xoff], message="This is xoff: ")

	yoff = tf.Print(yoff, [yoff], message="This is yoff: ")

	ch_shape = tf.shape(R)

	# get value at corner coordinates
	R1 = tf.reshape(get_val_at_coord(R,C1x,C1y), ch_shape)
	G1 = tf.reshape(get_val_at_coord(G,C1x,C1y), ch_shape)
	B1 = tf.reshape(get_val_at_coord(B,C1x,C1y), ch_shape)

	R2 = tf.reshape(get_val_at_coord(R,C2x,C2y), ch_shape)
	G2 = tf.reshape(get_val_at_coord(G,C2x,C2y), ch_shape)
	B2 = tf.reshape(get_val_at_coord(B,C2x,C2y), ch_shape)

	R3 = tf.reshape(get_val_at_coord(R,C3x,C3y), ch_shape)
	G3 = tf.reshape(get_val_at_coord(G,C3x,C3y), ch_shape)
	B3 = tf.reshape(get_val_at_coord(B,C3x,C3y), ch_shape)

	R4 = tf.reshape(get_val_at_coord(R,C4x,C4y), ch_shape)
	G4 = tf.reshape(get_val_at_coord(G,C4x,C4y), ch_shape)
	B4 = tf.reshape(get_val_at_coord(B,C4x,C4y), ch_shape)

	# interpolate
	Ri = 	(1-xoff)*(1-yoff)*R1 + \
			(xoff)	*(yoff)	 *R2 + \
			(1-yoff)*(xoff)	 *R3 + \
			(yoff)	*(1-xoff)*R4

	Gi = 	(1-xoff)*(1-yoff)*G1 + \
			(xoff)	*(yoff)	 *G2 + \
			(1-yoff)*(xoff)	 *G3 + \
			(yoff)	*(1-xoff)*G4

	Bi = 	(1-xoff)*(1-yoff)*B1 + \
			(xoff)	*(yoff)	 *B2 + \
			(1-yoff)*(xoff)	 *B3 + \
			(yoff)	*(1-xoff)*B4

	return tf.stack([Ri, Gi, Bi], axis=2)

def get_val_at_coord(im_channel, Cx, Cy):
	inp_shape = tf.shape(im_channel)
	im_channel_flat = tf.reshape(im_channel, [-1])
	Cx_flat = tf.cast(tf.reshape(Cx, [-1]), dtype=tf.int32)
	Cy_flat = tf.cast(tf.reshape(Cy, [-1]), dtype=tf.int32)
	ind_flat = (Cy_flat * inp_shape[1]) + Cx_flat

	return tf.gather(im_channel_flat, ind_flat)

