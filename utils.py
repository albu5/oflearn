import numpy as np
from skimage import io
from random import randint
from pipi.flow import readFlow


def translate(im, tx, ty, mode='edge'):
    im_ = np.lib.pad(im, ((abs(ty), abs(ty)), (abs(tx), abs(tx)), (0, 0)), mode)
    return im_[abs(ty)+ty:im.shape[0]+abs(ty)+ty][abs(tx)+tx:im.shape[1]+abs(tx)+tx][:]


def read_img(f_name):
    return io.imread(f_name).astype(np.float32) / 255


def get_batch(batch_size, img1_list, img2_list, edge_list, miss_list, flow_list, height, width):
    n_files = len(img2_list)
    imgs1 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    imgs2 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    flows = np.zeros((batch_size, height, width, 2), dtype=np.float32)
    edges = np.zeros((batch_size, height, width, 1), dtype=np.float32)
    misses = np.zeros((batch_size, height, width, 1), dtype=np.float32)

    good_ids = []

    for i in range(batch_size):
        try:
            idx = randint(0, n_files - 1)
            img1 = io.imread(img1_list[idx]).astype(np.float32) / 255
            img2 = io.imread(img2_list[idx]).astype(np.float32) / 255
            flow = (readFlow(flow_list[idx]).astype(np.float32))
            edge = np.expand_dims(io.imread(edge_list[idx])[:, :, 1].astype(np.float32) / 255, axis=2)
            miss = np.expand_dims(io.imread(miss_list[idx])[:, :, 1].astype(np.float32) / 255, axis=2)
            imgs1[i][:][:][:] = img1
            imgs2[i][:][:][:] = img2
            flows[i][:][:][:] = flow
            misses[i][:][:][:] = miss
            edges[i][:][:][:] = edge
            good_ids.append(True)
        except:
            good_ids.append(False)
    imgs1 = imgs1[good_ids][:][:][:]
    imgs2 = imgs2[good_ids][:][:][:]
    flows = flows[good_ids][:][:][:]
    edges = edges[good_ids][:][:][:]
    misses = misses[good_ids][:][:][:]
    # print(good_ids)
    return imgs1, imgs2, edges, misses, flows
