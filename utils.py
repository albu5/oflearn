import numpy as np
from skimage import io


def translate(im, tx, ty, mode='edge'):
    im_ = np.lib.pad(im, ((abs(ty), abs(ty)), (abs(tx), abs(tx)), (0, 0)), mode)
    return im_[abs(ty)+ty:im.shape[0]+abs(ty)+ty][abs(tx)+tx:im.shape[1]+abs(tx)+tx][:]


def read_img(f_name):
    return io.imread(f_name).astype(np.float32) / 255

