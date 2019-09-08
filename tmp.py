import tensorflow as tf
import numpy as np
import sys
from os import path
from datetime import datetime
from modules import *
from losses import *
from scipy.misc import imsave, imread

class FlowNetC(object):
    def __init__(self):
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        self.height = 224
        self.width = 224
        self.wlast = '/home/likewise-open/SENSETIME/qiudi/Documents/rgbd_processing_project/models/fullmodel/checkpoint/ckpt'
        # self.wlast = '/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/flownet-C.ckpt-0'
        self.L = tf.placeholder('float32', shape=[1, self.height, self.width, 3], name='i_L')
        self.Lw = tf.placeholder('float32', shape=[1, self.height, self.width, 3], name='i_Lw')
        self.R = tf.placeholder('float32', shape=[1, self.height, self.width, 3], name='i_R')
        
        self.feature_extract_init = FeatureExtractor()
        self.feature_extract_reuse = FeatureExtractor(reuse=True)
        self.flow_estimator_init = FlowEstimator(self.height, self.width)

        self._build_graph()
        all_var = tf.trainable_variables()
        self.load_saver_all = tf.train.Saver(all_var) 
        self.sess.run(tf.global_variables_initializer())

        self.load_saver_all.restore(self.sess, self.wlast)

    def _build_graph(self):
        L = tf.image.resize_bilinear(self.L, [384, 512])
        R = tf.image.resize_bilinear(self.R, [384, 512])
        featureL = self.feature_extract_init(L)
        featureR = self.feature_extract_reuse(R)
        output_rough = self.flow_estimator_init(featureL, featureR)
        self.rough_flow = output_rough['flow']
        self.flowL = inverse_warp_by_flow(self.Lw, self.rough_flow, self.height, self.width)
    

    def __call__(self, token):
        path = '/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/'
        R = imread(path + 'target2/' + token + '_target.png')/255.
        Lw = imread(path + 'train2/' + token + '_train.png')/255.
        L = np.mean(Lw, 2)
        L = np.stack([L, L, L], 2) * 4
        imsave('/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/warped_train2/' + token + '_train.png', Lw)
        imsave('/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/warped_train2/' + token + '_target.png', R)
        print(L.shape)
        L = np.expand_dims(L, 0)
        R = np.expand_dims(R, 0)
        Lw = np.expand_dims(Lw, 0)
        flowL, = self.sess.run([self.flowL], feed_dict={self.L : L, self.R : R, self.Lw : Lw})
        imsave('/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/warped_train2/' + token + '_warped_train.png', flowL[0, ...])
        

model = FlowNetC()
with open('/home/likewise-open/SENSETIME/qiudi/Downloads/Test_OpticalFlow/list.txt') as f:  
    for tk in f:
        tk = tk.rstrip('\n')
        model(tk)





        

