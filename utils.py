import numpy as np
from skimage import io
from scipy.misc import imresize
import time
from random import randint
from random import choice
from pipi.flow import readFlow
from scipy.ndimage.interpolation import zoom


def toss():
    return choice([True, False])


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


def get_batch_jitter(batch_size, img1_list, img2_list, edge_list, flow_list, height, width):
    n_files = len(img1_list)
    imgs1 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    imgs2 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    flows = np.zeros((batch_size, height, width, 2), dtype=np.float32)
    edges = np.zeros((batch_size, height, width, 1), dtype=np.float32)

    good_ids = []

    for i in range(batch_size):
        try:
            idx = randint(0, n_files - 1)
            img1 = io.imread(img1_list[idx]).astype(np.float32) / 255
            img2 = io.imread(img2_list[idx]).astype(np.float32) / 255
            flow = (readFlow(flow_list[idx]).astype(np.float32))
            edge = np.expand_dims(io.imread(edge_list[idx])[:, :, 1].astype(np.float32) / 255, axis=2)
            imgs1[i][:][:][:] = img1
            imgs2[i][:][:][:] = img2
            flows[i][:][:][:] = flow
            edges[i][:][:][:] = edge
            good_ids.append(True)
        except:
            good_ids.append(False)
    imgs1 = imgs1[good_ids][:][:][:]
    imgs2 = imgs2[good_ids][:][:][:]
    flows = flows[good_ids][:][:][:]
    edges = edges[good_ids][:][:][:]
    # print(good_ids)
    # apply horizontal flipping
    if toss():
        imgs1 = np.flip(imgs1, axis=2)
        imgs2 = np.flip(imgs2, axis=2)
        edges = np.flip(edges, axis=2)
        flows = np.flip(flows, axis=2)
        flows[:, :, :, 0] = -flows[:, :, :, 0]

    # apply vertical flipping
    if toss():
        imgs1 = np.flip(imgs1, axis=1)
        imgs2 = np.flip(imgs2, axis=1)
        edges = np.flip(edges, axis=1)
        flows = np.flip(flows, axis=1)
        flows[:, :, :, 1] = -flows[:, :, :, 1]

    return imgs1, imgs2, edges, flows


def get_batch_jitter_scale(batch_size, img1_list, img2_list, edge_list, miss_list, flow_list, height, width, scale=1):
    start_time = time.time()
    n_files = len(img1_list)
    imgs1 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    imgs2 = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    flows = np.zeros((batch_size, height, width, 2), dtype=np.float32)
    edges = np.zeros((batch_size, height, width, 1), dtype=np.float32)
    misses = np.zeros((batch_size, height, width, 1), dtype=np.float32)

    temp_f = np.zeros((height, width, 2), dtype=np.float32)

    good_ids = []

    for i in range(batch_size):

        try:
            idx = randint(0, n_files - 1)
            img1 = io.imread(img1_list[idx]).astype(np.float32) / 255
            img2 = io.imread(img2_list[idx]).astype(np.float32) / 255
            flow = readFlow(flow_list[idx]).astype(np.float32)
            edge = io.imread(edge_list[idx]).astype(np.float32) / 255
            miss = io.imread(miss_list[idx]).astype(np.float32) / 255
            jump = int(1/scale)
            if scale < 1:
                img1 = img1[::jump, ::jump, :]
                img2 = img2[::jump, ::jump, :]
                edge = edge[::jump, ::jump]
                miss = miss[::jump, ::jump]
                temp_f = flow[::jump, ::jump, :] * scale
            else:
                temp_f = flow

            edge = np.expand_dims(edge, axis=2)
            miss = np.expand_dims(miss, axis=2)
            imgs1[i][:][:][:] = img1
            imgs2[i][:][:][:] = img2
            flows[i][:][:][:] = temp_f
            edges[i][:][:][:] = edge
            misses[i][:][:][:] = miss
            good_ids.append(True)
        except:
            good_ids.append(False)
    imgs1 = imgs1[good_ids][:][:][:]
    imgs2 = imgs2[good_ids][:][:][:]
    flows = flows[good_ids][:][:][:]
    edges = edges[good_ids][:][:][:]
    misses = misses[good_ids][:][:][:]

    # print(good_ids)
    # apply horizontal flipping
    if toss():
        imgs1 = np.flip(imgs1, axis=2)
        imgs2 = np.flip(imgs2, axis=2)
        edges = np.flip(edges, axis=2)
        misses = np.flip(misses, axis=2)
        flows = np.flip(flows, axis=2)
        flows[:, :, :, 0] = -flows[:, :, :, 0]

    # apply vertical flipping
    if toss():
        imgs1 = np.flip(imgs1, axis=1)
        imgs2 = np.flip(imgs2, axis=1)
        edges = np.flip(edges, axis=1)
        misses = np.flip(misses, axis=1)
        flows = np.flip(flows, axis=1)
        flows[:, :, :, 1] = -flows[:, :, :, 1]
    elapsed_time = time.time() - start_time
    # print('time elapse in reading batch is', elapsed_time)
    return imgs1, imgs2, edges, misses, flows
