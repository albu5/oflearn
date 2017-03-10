import numpy as np

def translate(im, tx, ty,mode='edge'):
	im_ = np.lib.pad(im, ((abs(ty),abs(ty)),(abs(tx),abs(tx)),(0,0)),mode)
	return im_[abs(ty)+ty:im.shape[0]+abs(ty)+ty][abs(tx)+tx:im.shape[1]+abs(tx)+tx][:]