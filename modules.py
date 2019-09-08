'''
Building blocks implemented in Tensorflow

-Cross-modal flow estimation
    |-FeatureExtractor
    |-FlowEstimation
    |-pose_module 
        |-linear_pose_constraint
        |-pose_to_flow
    |-PoseFusion

-RGB-D module depth refinement
    |-RGBToFKPN


Di Qiu, 13-06-2019
'''
import numpy as np
import tensorflow as tf
from utils.utils import LeakyReLU, pad, antipad
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from utils.correlation import correlation
from utils.camera_util import *
from utils.flowlib import *
slim = tf.contrib.slim


##### Start of cross-modal flow estimation #####

class FeatureExtractor(object):
    '''
    Feature extraction tower for FlowNetC
    FlowNetC implementation taken from: https://github.com/sampepose/flownet2-tf
    '''
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self, input):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            # We will do our own padding to match the original Caffe code
                            padding='VALID'):

            weights_regularizer = slim.l2_regularizer(0.0004)
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.conv2d], stride=2):
                    conv_a_1 = slim.conv2d(pad(input, 3), 64, 7, scope='FlowNetC/conv1', reuse=self.reuse)
                    conv_a_2 = slim.conv2d(pad(conv_a_1, 2), 128, 5, scope='FlowNetC/conv2', reuse=self.reuse)
                    conv_a_3 = slim.conv2d(pad(conv_a_2, 2), 256, 5, scope='FlowNetC/conv3', reuse=self.reuse)

        return {'conv_1': conv_a_1,
                'conv_2': conv_a_2,
                'conv_3': conv_a_3
                }


class FlowEstimator(object): 
    '''
    FlowNetC implementation taken from: https://github.com/sampepose/flownet2-tf
    '''
    def __init__(self, height, width, reuse=False):
        self.reuse = reuse
        self.height = height
        self.width = width

    def __call__(self, featureA, featureB):
        conv_a_2 = featureA['conv_2']
        conv_a_3 = featureA['conv_3']
        conv_b_3 = featureB['conv_3']
        # print(conv_a_3.get_shape().as_list())
        # print(conv_b_3.get_shape().as_list())
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            # We will do our own padding to match the original Caffe code
                            padding='VALID'):

            cc = correlation(conv_a_3, conv_b_3, 1, 20, 1, 2, 20)
            cc_relu = LeakyReLU(cc)

            # Combine cross correlation results with convolution of feature map A
            net_conv = slim.conv2d(conv_a_3, 32, 1, scope='FlowNetC/conv_redir', reuse=self.reuse)
            # Concatenate along the channels axis
            net = tf.concat([net_conv, cc_relu], axis=3)

            conv3_1 = slim.conv2d(pad(net), 256, 3, scope='FlowNetC/conv3_1', reuse=self.reuse)
            # print(conv3_1.get_shape().as_list())
            with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3, reuse=self.reuse):
                conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='FlowNetC/conv4')

                conv4_1 = slim.conv2d(pad(conv4), scope='FlowNetC/conv4_1')

                conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='FlowNetC/conv5')
                conv5_1 = slim.conv2d(pad(conv5), scope='FlowNetC/conv5_1')

            conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='FlowNetC/conv6', reuse=self.reuse)
            # print(conv6.get_shape().as_list())
            conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='FlowNetC/conv6_1', reuse=self.reuse)

            """ START: Refinement Network """
            with slim.arg_scope([slim.conv2d], reuse=self.reuse):
                with slim.arg_scope([slim.conv2d_transpose], biases_initializer=None, reuse=self.reuse):
                    predict_flow6 = slim.conv2d(pad(conv6_1), 2, 3,
                                                scope='FlowNetC/predict_flow6',
                                                activation_fn=None)
                    deconv5 = antipad(slim.conv2d_transpose(conv6, 512, 4,
                                                            stride=2,
                                                            scope='FlowNetC/deconv5'))
                    upsample_flow6to5 = antipad(slim.conv2d_transpose(predict_flow6, 2, 4,
                                                                      stride=2,
                                                                      scope='FlowNetC/upsample_flow6to5',
                                                                      activation_fn=None))
                    concat5 = tf.concat([conv5_1, deconv5, upsample_flow6to5], axis=3)

                    predict_flow5 = slim.conv2d(pad(concat5), 2, 3,
                                                scope='FlowNetC/predict_flow5',
                                                activation_fn=None)
                    deconv4 = antipad(slim.conv2d_transpose(concat5, 256, 4,
                                                            stride=2,
                                                            scope='FlowNetC/deconv4'))
                    upsample_flow5to4 = antipad(slim.conv2d_transpose(predict_flow5, 2, 4,
                                                                      stride=2,
                                                                      scope='FlowNetC/upsample_flow5to4',
                                                                      activation_fn=None))
                    concat4 = tf.concat([conv4_1, deconv4, upsample_flow5to4], axis=3)

                    predict_flow4 = slim.conv2d(pad(concat4), 2, 3,
                                                scope='FlowNetC/predict_flow4',
                                                activation_fn=None)
                    deconv3 = antipad(slim.conv2d_transpose(concat4, 128, 4,
                                                            stride=2,
                                                            scope='FlowNetC/deconv3'))
                    upsample_flow4to3 = antipad(slim.conv2d_transpose(predict_flow4, 2, 4,
                                                                      stride=2,
                                                                      scope='FlowNetC/upsample_flow4to3',
                                                                      activation_fn=None))
                    concat3 = tf.concat([conv3_1, deconv3, upsample_flow4to3], axis=3)

                    predict_flow3 = slim.conv2d(pad(concat3), 2, 3,
                                                scope='FlowNetC/predict_flow3',
                                                activation_fn=None)
                    deconv2 = antipad(slim.conv2d_transpose(concat3, 64, 4,
                                                            stride=2,
                                                            scope='FlowNetC/deconv2'))
                    upsample_flow3to2 = antipad(slim.conv2d_transpose(predict_flow3, 2, 4,
                                                                      stride=2,
                                                                      scope='FlowNetC/upsample_flow3to2',
                                                                      activation_fn=None))
                    concat2 = tf.concat([conv_a_2, deconv2, upsample_flow3to2], axis=3)

                    predict_flow2 = slim.conv2d(pad(concat2), 2, 3,
                                                scope='FlowNetC/predict_flow2',
                                                activation_fn=None)
            """ END: Refinement Network """

            flow = predict_flow2
            # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
            flow = tf.image.resize_bilinear(flow,
                                            tf.stack([self.height, self.width]),
                                            align_corners=True)

        return {
                'predict_flows': [predict_flow6, predict_flow5, predict_flow4, predict_flow3, predict_flow2],
                'flow': flow,
                'feature': conv3_1
                }


def pose_module(D, rf_flow, conf_):
    '''

    :param D:
    :param rf_predictions:
    :param conf_:
    :return:
    '''

    b_, h_, w_, _ = D.get_shape().as_list()
    pose = linear_pose_constraint(D, rf_flow, conf_)
    pose_flow = pose_to_flow(pose, D)

    return pose_flow, pose


def linear_pose_constraint(D, flow, weight):
    # build left hand matrix
    nD = D/1000. # some scale change
    nD = tf.squeeze(nD)
    b, h, w, _ = flow.get_shape().as_list()
    D_vec = tf.reshape(nD, [b, h*w, 1])
    weight = tf.reshape(weight, [b, h*w, 1])
    M = tf.concat([D_vec, tf.ones([b, h*w, 1], 'float32')], 2) * weight
    # build right hand side
    dflowx = nD * flow[:, :, :, 0]
    dflowy = nD * flow[:, :, :, 1]
    dflowx = tf.reshape(dflowx, [b, h*w, 1]) * weight
    dflowy = tf.reshape(dflowy, [b, h*w, 1]) * weight
    rhs = tf.concat([dflowx, dflowy], 2)
    sol = tf.matrix_solve_ls(M/10000., rhs/10000., l2_regularizer=0.00001) # some other stupid redundant scale change...

    return {'c': sol[:, 0, :], 't': sol[:, 1, :] * 1000.}


def pose_to_flow(pose, D):
    b_, h_, w_, _ = D.get_shape().as_list()

    c = pose['c']
    c = tf.reshape(c, [b_, 1, 1, 2])
    t = pose['t']
    t = tf.reshape(t, [b_, 1, 1, 2])
    pose_flow = c + t / D

    return pose_flow

class PoseFusion(object):
    def __init__(self, reuse=False):
        self.reuse = reuse

    def __call__(self,  pose_flow, flow):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=LeakyReLU,
                            weights_regularizer=slim.l2_regularizer(0.0004)):
            inputs = tf.concat([pose_flow, flow], 3)
            conv0 = slim.conv2d(inputs, 64, 3, stride=1, scope='pose/conv0', padding='SAME')
            conv1 = slim.conv2d(conv0, 64, 3, stride=2, scope='pose/conv1', padding='SAME')
            conv1_1 = slim.conv2d(conv1, 128, 3, stride=1, scope='pose/conv1_1', padding='SAME')
            conv2 = slim.conv2d(conv1_1, 128, 3, stride=2, scope='pose/conv2', padding='SAME')
            conv2_1 = slim.conv2d(conv2, 128, 3, stride=1, scope='pose/conv2_1', padding='SAME')
            predict_flow2 = slim.conv2d(conv2_1, 2, 3, stride=1, scope='pose/predict_flow2', padding='SAME')

            upconv1 = slim.conv2d_transpose(tf.concat([predict_flow2, conv2_1], 3), 128, 4, stride=2, scope='pose/upconv1', padding='SAME')
            _, h1_, w1_, _ = upconv1.get_shape().as_list()
            rconv1 = slim.conv2d(tf.concat([upconv1, conv1_1], 3), 64, 3,
                                 stride=1, scope='pose/rconv1', padding='SAME')
            predict_flow1 = slim.conv2d(rconv1, 2, 3, stride=1, scope='pose/predict_flow1', padding='SAME')
            upconv0 = slim.conv2d_transpose(tf.concat([predict_flow1, rconv1] ,3), 64, 4, stride=2, scope='pose/upconv0', padding='SAME')
            _, h0_, w0_, _ = upconv0.get_shape().as_list()
            rconv0 = slim.conv2d(tf.concat([upconv0, conv0], 3), 64, 3,
                                 stride=1, scope='pose/rconv0', padding='SAME')
            predict_flow0 = slim.conv2d(rconv0, 2, 3, stride=1, scope='pose/predict_flow0', padding='SAME')

            return [predict_flow2, predict_flow1, predict_flow0]

##### End of cross-modal flow estimation #####


##### Start of RGB-D depth refinement #####

class RGBToFKPN(object):
    def __init__(self, ks=3):
        self.ks = ks
    def __call__(self, flowD, R, flowL):
        nch = 64
        nrep = 1
        ks = self.ks

        R = R - tf.reduce_mean(R, [1, 2, 3], keep_dims=True)
        flowL = tf.tile(flowL, [1, 1, 1, 3])
        inputs = tf.concat([flowD, flowL, R], 3)
        #inputs = tf.concat([flowD, flowL], 3)
        _, height, width, _ = flowD.get_shape().as_list()
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),  # biases_initializer=None,
                            # normalizer_fn=slim.batch_norm,
                            # tf.truncated_normal_initializer(stddev=0.1), #slim.variance_scaling_initializer(),
                            # tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32),
                            activation_fn=LeakyReLU,
                            padding='SAME'):
            # weights_regularizer=slim.l2_regularizer(0.00004)):
            with tf.variable_scope('dynamic'):
                dconv0 = slim.conv2d(inputs, nch, 3, stride=1, scope='ddk/conv0', padding='SAME')
                dconv1 = slim.repeat(dconv0, nrep, slim.conv2d, nch, [3, 3], stride=1, scope='ddk/conv1', padding='SAME')
                # print(dconv1.get_shape().as_list())
                dconv2 = slim.conv2d(dconv1, nch * 2, 3, stride=2, scope='ddk/conv2', padding='SAME')
                dconv2_1 = slim.repeat(dconv2, nrep, slim.conv2d, nch * 2, [3, 3], stride=1, scope='ddk/conv2_1',
                                    padding='SAME')
                # print(dconv2_1.get_shape().as_list())
                dconv3 = slim.conv2d(dconv2_1, nch * 2, 3, stride=2, scope='ddk/conv3', padding='SAME')
                dconv3_1 = slim.repeat(dconv3, nrep, slim.conv2d, nch * 2, [3, 3], stride=1, scope='ddk/conv3_1',
                                    padding='SAME')
                # print(dconv3_1.get_shape().as_list())
                dconv4 = slim.conv2d(dconv3_1, nch * 4, 3, stride=2, scope='ddk/conv4', padding='SAME')
                dconv4_1 = slim.repeat(dconv4, nrep, slim.conv2d, nch * 4, [3, 3], stride=1, scope='ddk/conv4_1',
                                    padding='SAME')
                # print(dconv4_1.get_shape().as_list())

                # UP 1
                ddeconv_n1 = slim.conv2d_transpose(tf.concat([dconv4_1], 3), nch * 2, 3, stride=2,
                                                scope='ddk/deconv_n1', padding='SAME')
                ddeconv_n1_1 = slim.repeat(tf.concat([ddeconv_n1], 3), nrep, slim.conv2d, nch * 2, [3, 3], stride=1,
                                        scope='ddk/deconv_n1_1', padding='SAME')
                # print(ddeconv_n1_1.get_shape().as_list())
                #

                # UP 2
                ddeconv_n2 = slim.conv2d_transpose(tf.concat([ddeconv_n1_1, dconv3_1], 3), nch * 2, 3, stride=2,
                                                scope='ddk/deconv_n2', padding='SAME')
                ddeconv_n2_1 = slim.repeat(tf.concat([ddeconv_n2], 3), nrep, slim.conv2d, nch * 2, [3, 3], stride=1,
                                        scope='ddk/deconv_n2_1', padding='SAME')
                # print(ddeconv_n2_1.get_shape().as_list())
                # pwd2 = slim.conv2d_transpose(ddeconv_n2_1, nch, 3, stride=2,

                # UP 3
                ddeconv_n3 = slim.conv2d_transpose(tf.concat([ddeconv_n2_1, dconv2_1], 3), nch, 3, stride=2,
                                                scope='ddk/deconv_n3', padding='SAME')
                ddeconv_n3_1 = slim.repeat(tf.concat([dconv1, ddeconv_n3], 3), nrep, slim.conv2d, nch, [3, 3], stride=1,
                                        scope='ddk/deconv_n3_1', padding='SAME')
                # print(ddeconv_n3_1.get_shape().as_list())

                # Apply kpn
                wb = slim.repeat(ddeconv_n3_1, nrep, slim.conv2d, ks ** 2 + 1, [3, 3], scope='ddk/w', padding='SAME',
                                biases_initializer=None, activation_fn=None)
                w = wb[:, :, :, :-1]
                b = tf.expand_dims(wb[:, :, :, -1], 3)
                w = w / tf.reduce_sum(tf.abs(w) + 1e-6, 3, keep_dims=True)

                img_patch = tf.extract_image_patches(flowD+b , [1, ks, ks, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
                filtimg = tf.reduce_sum((img_patch) * w, 3, keep_dims=True)
                # filtimg = tf.image.resize_bilinear(pfiltimg3, [H, W])

            return [filtimg, w, b]  # [dx, dy, w, b]

##### End of RGB-D module depth refinement #####

