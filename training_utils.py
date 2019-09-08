'''

'''
import tensorflow as tf
import numpy as np

def visualize_kernel(tensor, ind):
    vis = tf.gather_nd(tensor, ind)
    H, W = vis.get_shape().as_list()
    vis = tf.expand_dims(vis, 0)
    vis = tf.expand_dims(vis, 3)

    vis = tf.image.resize_nearest_neighbor(vis, [10*H, 10*W])

    return vis

def get_vis_ind(H, W, ks, str):

    h = H // str
    w = W // str
    print([h, w, 4])
    ind = np.zeros([h * ks, w * ks, 4], np.int)
    for i in range(0, h): # big row
        for j in range(0, w): # big col
            for f in range(0, ks): # small row
                for k in range(0, ks): # small col
                    ind[i*ks + f, j*ks + k, 0] = 0
                    ind[i*ks + f, j*ks + k, 1] = i*str
                    ind[i*ks + f, j*ks + k, 2] = j*str
                    ind[i*ks + f, j*ks + k, 3] = f*ks + k
    return ind

