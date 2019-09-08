'''
Loss function & Evaluation metrics implementations in TensorFlow and numpy

@Di Qiu, 23-07-2019
'''

import tensorflow as tf
import numpy as np

def L1loss(x, y): # shape(# batch, h, w, 2)
    # return tf.reduce_mean(tf.reduce_sum(tf.pow(tf.abs(x-y), 1), axis=3))
    return tf.reduce_mean(tf.pow(tf.abs(x-y), 1))


def L2loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.pow(tf.abs(x-y), 2))


# end point error
def EPE(flows_gt, flows):
    return tf.reduce_mean(tf.norm(flows_gt-flows, ord = 2, axis = 3))


def multiscale_lossL2(flows_gt, flows_pyramid, conf,
                    weights, name = 'multiscale_loss'):
    with tf.name_scope(name) as ns:
        # Calculate mutiscale loss
        loss = 0.
        for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
            # Downsampling the scaled ground truth flow
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt, (h, w))
            fs_conf = tf.image.resize_bilinear(conf, (h, w))
            # Calculate l2 loss
            loss += weight*L2loss(fs_gt_down * fs_conf, fs * fs_conf)

        return loss


def multiscale_lossL1(flows_gt, flows_pyramid, conf,
                    weights, name='multiscale_lossL1'):
    with tf.name_scope(name) as ns:
        # Calculate mutiscale loss
        loss = 0.
        for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
            # Downsampling the scaled ground truth flow
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt, (h, w))
            fs_conf = tf.image.resize_bilinear(conf, (h, w))
            # Calculate l2 loss
            loss += weight*L1loss(fs_gt_down * fs_conf, fs * fs_conf)

        return loss


def multirobust_loss(flows_gt, flows_pyramid,
                     weights, epsilon = 0.01,
                     q = 0.4, name = 'multirobust_loss'):
    with tf.name_scope(name) as ns:
        loss = 0.
        for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
            # Downsampling the scaled ground truth flow
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt, (h, w))
            # Calculate l1 loss
            _l = L1loss(fs_gt_down, fs)
            loss += weight*(_l+epsilon)**q

    return loss


def sobel_edges(img):
    ch = img.get_shape().as_list()[3]
    kerx = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    kery = tf.constant([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype='float32')
    kerx = tf.expand_dims(kerx, 2)
    kerx = tf.expand_dims(kerx, 3)
    kerx = tf.tile(kerx, [1, 1, ch, 1])
    kery = tf.expand_dims(kery, 2)
    kery = tf.expand_dims(kery, 3)
    kery = tf.tile(kery, [1, 1, ch, 1])
    gx = tf.nn.depthwise_conv2d_native(img, kerx, strides=[1,1,1,1], padding="VALID")
    gy = tf.nn.depthwise_conv2d_native(img, kery, strides=[1,1,1,1], padding="VALID")
    return tf.concat([gx, gy], 3)


def sobel_gradient_loss(guess, truth):
    g1 = sobel_edges(guess)
    g2 = sobel_edges(truth)
    return tf.reduce_mean(tf.pow(tf.abs(g1 - g2),1))


def quantile_loss(guess, input, truth, conf, a=25, b=50, c=75):
    diff = np.abs(input*conf - truth*conf)
    diff2 = np.abs(guess - truth)
    diff3 = diff[np.nonzero(diff)]
    qa, qb, qc = np.percentile(diff3, [a, b, c])
    maska = np.less_equal(diff, qa).astype(np.float32)*conf
    maskb2 = np.less_equal(diff, qb).astype(np.float32)*conf
    maskc2 = np.less_equal(diff, qc).astype(np.float32)*conf
    maskb = maskb2 - maska
    maskc = maskc2 - maskb2
    maska = maska / (np.sum(maska) + 1e-6)
    maskb = maskb / (np.sum(maskb) + 1e-6)
    maskc = maskc / (np.sum(maskc) + 1e-6)
    lossa = np.sum(diff2 * maska)
    lossb = np.sum(diff2 * maskb)
    lossc = np.sum(diff2 * maskc)

    return lossa, lossb, lossc


