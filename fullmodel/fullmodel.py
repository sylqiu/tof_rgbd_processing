'''
ToF RGB-D Alignment & Refinement TensorFlow implementation

@Di Qiu, 23-07-2019
'''
import time
import threading
import os
import argparse
import tensorflow as tf
import numpy as np
import sys
from os import path
from datetime import datetime
sys.path.insert(0, '../..')
from loaders import *
from loader_utils import *
from modules import *
from losses import *
from utils.logging_utils import *
from scipy.misc import imsave


class TrainingConfig(object):
    def __init__(self):
        self.wlast = ''
        self.save_model_dir = './checkpoint/'
        self.l_rate = 1*1e-5
        self.total_iter = 10000
        self.summary_dir = './'
        self.display_step = 100
        self.snapshot = 5000
        self.is_from_scratch = False
        self.bs = 2
        self.path = r''
        self.training_decay_step = 10000
        self.decay_rate = 0.7


class TestingConfig(object):
    def __init__(self, dataset_name):
        
        self.wlast = '../../checkpoint/ckpt'
        self.output_save_dir ='./results'
        self.bs = 1

        #dataset path
        self.path = ''
        self.use_fifo = False


class Model(object):  
    def __init__(self, 
                    dataset_name, 
                    dataset2_name, 
                    sigma, 
                    is_training, 
                    mvg_aug,
                    is_from_scratch,
                    pretrain_ckpt_path):
        '''
        sigma: perturbation strength for testing flow estimation
        mvg_aug: whether to use online warping, must be True if sigma>0
        '''
        self.dataset_handle = get_dataset(dataset_name)
        self.secondary_dataset = get_dataset(dataset2_name)
        self.sigma = sigma
        self.gt_flag = True
        self.is_training = is_training
        self.mvg_aug = mvg_aug

        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

        # original image size
        self.ori_height = 480
        self.ori_width = 640

        # cropping image size
        self.height = 384
        self.width = 512
          
        if is_training:
            self.crop_type = 'random'
            self.training_config = TrainingConfig()

            if is_from_scratch == 0:
                self.training_config.is_from_scratch = False
                self.training_config.wlast = pretrain_ckpt_path

            self._build_loader(self.training_config.path, split='train', mvg_aug=mvg_aug)
            self.bs = self.training_config.bs
            self._declare_placeholders()
            self._build_graph()
            self._collect_variables()
            self._build_optimization()
            self._build_summary()
            self.idxtype = 'random'
        else:
            self.crop_type = None
            
            self.testing_config = TestingConfig()
            self.use_fifo = self.testing_config.use_fifo

            if pretrain_ckpt_path != 'none':
                self.testing_config.wlast = pretrain_ckpt_path

            self._build_loader(self.testing_config.path, split='test', mvg_aug=mvg_aug)
            self.bs = self.testing_config.bs
            self._declare_placeholders()
            self._build_graph()
            self._collect_variables()
            self.idxtype = None

        self.sess.run(tf.global_variables_initializer())
    

    def _declare_placeholders(self):
        self.i_D = tf.placeholder('float32', shape=[self.height, self.width, 1], name='i_D')
        self.i_L = tf.placeholder('float32', shape=[self.height, self.width, 1], name='i_L')
        self.i_R = tf.placeholder('float32', shape=[self.height, self.width, 3], name='i_R')
        self.i_conf = tf.placeholder('float32', shape=[self.height, self.width, 1], name='i_conf')


        
        if self.mvg_aug: # need to warp
            self.i_gt_D = tf.placeholder('float32', shape=[self.height, self.width, 1], name='i_gt_D')
            self.i_gt_invflow = tf.placeholder('float32', shape=[self.height, self.width, 2], name='gt_invflow')

            self.i_R_orisize = tf.placeholder(tf.float32, shape=[self.ori_height, self.ori_width, 3], name='R_orisize')
            self.i_invflow_orisize = tf.placeholder(tf.float32, shape=[self.ori_height, self.ori_width, 2], name='invflow_orisize')
            self.i_conf_orisize = tf.placeholder(tf.float32, shape=[self.ori_height, self.ori_width, 1], name='conf_orisize')
            self.i_gt_D_orisize = tf.placeholder(tf.float32, shape=[self.ori_height, self.ori_width, 1], name='gt_D_orisize')

            self.warped_R = inverse_warp_by_flow(tf.expand_dims(self.i_R_orisize, 0),
                                            tf.expand_dims(self.i_invflow_orisize, 0), self.ori_height, self.ori_width)
            self.warped_conf = inverse_warp_by_flow(tf.expand_dims(self.i_conf_orisize, 0),
                                            tf.expand_dims(self.i_invflow_orisize, 0), self.ori_height, self.ori_width)
            self.warped_D = inverse_warp_by_flow(tf.expand_dims(self.i_gt_D_orisize, 0),
                                            tf.expand_dims(self.i_invflow_orisize, 0), self.ori_height, self.ori_width)
            
            queue = tf.FIFOQueue(40, 
                                ['float32' for _ in range(6)],
                                shapes=[
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 3],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 2]
                                ])
            self.enqueue_op = queue.enqueue([self.i_D, self.i_L, self.i_R, self.i_conf, self.i_gt_D, self.i_gt_invflow])
            self.D, self.L, self.R, self.conf, self.gt_D, self.gt_invflow = queue.dequeue_many(self.bs)
            
        elif self.gt_flag: # testing, no need to warp but has to read gt flow
            self.i_gt_D = tf.placeholder('float32', shape=[self.height, self.width, 1], name='i_gt_D')
            self.i_gt_invflow = tf.placeholder('float32', shape=[self.height, self.width, 2], name='gt_invflow')
            
            queue = tf.FIFOQueue(40, 
                                ['float32' for i in range(6)],
                                shapes=[
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 3],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 1],
                                    [self.height, self.width, 2]
                                ])
            self.enqueue_op = queue.enqueue([self.i_D, self.i_L, self.i_R, self.i_conf, self.i_gt_D, self.i_gt_invflow])
            self.D, self.L, self.R, self.conf, self.gt_D, self.gt_invflow = queue.dequeue_many(self.bs)

        else: # no ground truth deployment
            if self.use_fifo = True:
                queue = tf.FIFOQueue(40, 
                                    ['float32' for i in range(4)],
                                    shapes=[
                                        [self.height, self.width, 1],
                                        [self.height, self.width, 1],
                                        [self.height, self.width, 3],
                                        [self.height, self.width, 1]
                                    ])
                self.enqueue_op = queue.enqueue([self.i_D, self.i_L, self.i_R, self.i_conf])
                self.D, self.L, self.R, self.conf = queue.dequeue_many(self.bs)
            else:
                self.D = tf.placeholder('float32', shape=[1, self.height, self.width, 1], name='i_D')
                self.L = tf.placeholder('float32', shape=[1, self.height, self.width, 1], name='i_L')
                self.R = tf.placeholder('float32', shape=[1, self.height, self.width, 3], name='i_R')
                self.conf = tf.placeholder('float32', shape=[1, self.height, self.width, 1], name='i_conf')


    def _build_loader(self, path, split, mvg_aug, align_flag=False):

        self.dataset = self.dataset_handle(path_to_data=path, sigma=self.sigma, 
                                            split=split,
                                            mvg_aug=mvg_aug, align_flag=align_flag)
        if self.dataset.name == 'nogt':
            assert self.is_training == 0
            self.gt_flag = False
        self.dataset_len = len(self.dataset)
        print('************** Using primary dataset {} **************'.format(self.dataset.name))

        if self.secondary_dataset is not None and self.is_training == 1:
            self.dataset2 = self.secondary_dataset(path_to_data=path, 
                                split=split, mvg_aug=mvg_aug, align_flag=align_flag
                                            ) 
            print('************** Using secondary dataset {} **************'.format(self.dataset2.name))
        
    

    def _build_graph(self):

        ## Alignment module
        self.pose_fuse = PoseFusion()
        self.feature_extract_init = FeatureExtractor()
        self.feature_extract_reuse = FeatureExtractor(reuse=True)
        self.flow_estimator_init = FlowEstimator(self.height, self.width)

        ## KPN module
        self.kpn = RGBToFKPN()

        ### Forward begins ###

        ## Alignment
        featureL = self.feature_extract_init(tf.tile(self.L, [1, 1, 1, 3]))
        featureR = self.feature_extract_reuse(self.R)
        output_rough = self.flow_estimator_init(featureL, featureR)
        self.rough_flow = output_rough['flow']

        ## Adjust confidence after rough flow estimation
        flowD1 = inverse_warp_by_flow(self.D, self.rough_flow, self.height, self.width)
        flowD1_where = tf.cast(tf.less_equal(flowD1*4095, 100), 'float32')
        flowD1 = (1 - flowD1_where) * flowD1 + (flowD1_where)
        flowconf1 = inverse_warp_by_flow(self.conf, self.rough_flow, self.height, self.width)
        flowconf1 = flowconf1 * (1. - flowD1_where)

        ## Solve linear system
        pose_flow, _ = pose_module(flowD1*4095, self.rough_flow, flowconf1) # comment this if direct fusion ablation

        ## Flow fusion
        # output_posefuse = self.pose_fuse(flowD1, self.rough_flow) # this is direct fusion ablation, not perfoming well
        output_posefuse = self.pose_fuse(pose_flow, self.rough_flow)
        posefuse_flow = output_posefuse[2]
        self.refine_flow = posefuse_flow

        ## Warp images and ready for KPN
        self.flowL = inverse_warp_by_flow(self.L, posefuse_flow, self.height, self.width)
        self.flowD2 = inverse_warp_by_flow(self.D, posefuse_flow, self.height, self.width)
        flowconf2 = inverse_warp_by_flow(self.conf, posefuse_flow, self.height, self.width)
        flowD2_where = tf.maximum(tf.cast(tf.less_equal(self.flowD2*4095, 5), 'float32'), \
                       tf.cast(tf.less_equal(flowconf2, 0.5), 'float32'))
        flowconf2 = flowconf2 * (1 - flowD2_where)
        conf = flowconf2 * self.conf
        self.filtered_D, self.w, self.b = self.kpn(self.flowD2, self.R, self.flowL)

        ### Forward Ends ###

        # Compute losses if ground truth is available
        if self.gt_flag:
            self.loss_roughflow = multiscale_lossL1(self.gt_invflow, output_rough['predict_flows'], conf,\
                                          weights=[0.32, 0.08, 0.02, 0.01, 0.005],
                                          name='loss_flow')
            self.loss_refineflow = multiscale_lossL1(self.gt_invflow, output_posefuse, conf,\
                                             weights=[0.02, 0.01, 0.005],
                                             name='loss_posefuse')

            self.roughflow_EPE = EPE(self.gt_invflow * self.conf, self.rough_flow * self.conf) * (384*512/tf.reduce_sum(self.conf))
            self.refineflow_EPE = EPE(self.gt_invflow * self.conf, self.refine_flow * self.conf) * (384*512/tf.reduce_sum(self.conf))
            self.L1depth = L1loss(self.filtered_D * conf, self.gt_D * conf)
            self.MAEdepth = L1loss(self.filtered_D * self.conf, self.gt_D * self.conf) * (384*512/tf.reduce_sum(self.conf))
            self.L1grad_depth = sobel_gradient_loss(self.filtered_D * conf, self.gt_D * conf) 

            self.loss_optimize = self.loss_roughflow + self.loss_refineflow +\
                        self.L1depth + 10.* self.L1grad_depth  
            


    def _collect_variables(self):
        all_var = tf.trainable_variables()
        var_flow = [var for var in all_var if 'FlowNetC' in var.name]
        var_pose = [var for var in all_var if 'pose' in var.name]
        var_kpn = [var for var in all_var if 'ddk' in var.name]
        self.var_opt = all_var

        self.model_saver = tf.train.Saver(all_var)

        self.load_saver_all = tf.train.Saver(all_var)
        self.load_saver_ddk = tf.train.Saver(var_kpn)
        self.load_saver_flow = tf.train.Saver(var_flow)
        self.load_saver_pose = tf.train.Saver(var_pose)



    def _build_summary(self):
        assert self.is_training == 1
        self.out_flow_img1 = tf.py_func(flow_to_image, [self.rough_flow[0, :, :, :] * 20], tf.uint8)
        self.out_flow_img1 = tf.expand_dims(self.out_flow_img1, 0)
        self.out_flow_img2 = tf.py_func(flow_to_image, [self.refine_flow[0, :, :, :] * 20], tf.uint8)
        self.out_flow_img2 = tf.expand_dims(self.out_flow_img2, 0)
        self.gt_invflowimg = tf.py_func(flow_to_image, [self.gt_invflow[0, :, :, :] * 20], tf.uint8)
        self.gt_invflowimg = tf.expand_dims(self.gt_invflowimg, 0)

        self.writer = tf.summary.FileWriter(self.training_config.summary_dir, self.sess.graph)
        tf.summary.image('D_ori', self.D[0, None, ...])
        tf.summary.image('RGB', self.R[0, None, ...])
        tf.summary.image('IR', self.L[0, None, ...])
        tf.summary.image('filtered_D', self.filtered_D[0, None, ...])
        tf.summary.image('gtD', self.gt_D[0, None, ...])
        tf.summary.image('conf', self.conf[0, None, ...])
        tf.summary.scalar('loss', self.loss_optimize)
        # tf.summary.image('before_pose', self.out_flow_img1)
        # tf.summary.image('after_pose', self.out_flow_img2)
        # tf.summary.scalar('flow_loss1', self.loss_flow)
        # tf.summary.scalar('flow_loss2', self.loss_posefuse)
        
        # tf.summary.image('gtflow_img', self.gt_invflowimg)

        with tf.variable_scope('summary'):
            self.summary_op = tf.summary.merge_all()
        


    def _build_optimization(self):
        
        step_now = tf.Variable(0, trainable=False)
        l_rate_decay1 = tf.train.exponential_decay(self.training_config.l_rate, step_now, \
                        self.training_config.training_decay_step, self.training_config.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(l_rate_decay1, epsilon=1e-4)
        
        grads = self.optimizer.compute_gradients(self.loss_optimize, var_list=self.var_opt)
        crap = [(grad, var) for grad, var in grads]
        self.trainstep = self.optimizer.apply_gradients(crap, global_step=step_now)


    def _fifo(self):
        idx = 0
        while True:
            if self.mvg_aug: # perform warping
                if self.idxtype == 'random': # random sampling from data
                    if self.secondary_dataset is not None:
                        coin = np.random.uniform(0, 1)
                        if coin < 0.9:
                            data_in = self.dataset.next_random_sample()
                        elif coin >= 0.9:
                            data_in = self.dataset2.next_random_sample()
                    else:
                        data_in = self.dataset.next_random_sample()

                else: # loop over the primary dataset
                    data_in = self.dataset.get_data(idx)
                    idx += 1
                    idx = idx % self.dataset_len

                imgR, img_confR, wD = self.sess.run([self.warped_R, self.warped_conf, self.warped_D],
                                                feed_dict={self.i_R_orisize: data_in['R_ori'],
                                                        self.i_invflow_orisize: data_in['gt_invflow'],
                                                        self.i_conf_orisize: data_in['conf'],
                                                        self.i_gt_D_orisize: data_in['gt_D']})
                imgR = imgR[0, :, :, :]
                img_confR = img_confR[0, :, :, :]
                wD = wD[0, :, :, :]
                cropped_data_in = crop_multiple_imgs(self.height, self.width, 
                                                    self.ori_height, self.ori_width,
                                                    self.crop_type, 10, 10,
                                                    data_in['D_ori'], img_confR,
                                                    wD, imgR,
                                                    data_in['L'], data_in['gt_invflow']
                                                    ) 
                self.sess.run(self.enqueue_op,
                        feed_dict={
                            self.i_D: cropped_data_in[0],
                            self.i_conf: cropped_data_in[1],
                            self.i_gt_D: cropped_data_in[2],
                            self.i_R: cropped_data_in[3],
                            self.i_L: cropped_data_in[4],
                            self.i_gt_invflow: cropped_data_in[5]
                        })         


 
            elif self.gt_flag: # no warping but read ground truth flow
                assert self.idxtype == None # only do this in testing mode
                data_in = self.dataset.get_data(idx)
                cropped_data_in = crop_multiple_imgs(self.height, self.width, 
                                                    self.ori_height, self.ori_width,
                                                    self.crop_type, 10, 10,
                                                    data_in['D_ori'], data_in['conf'],
                                                    data_in['gt_D'], data_in['R_ori'],
                                                    data_in['L'], data_in['gt_invflow']
                                                    )
                self.sess.run(self.enqueue_op,
                              feed_dict={
                                  self.i_D: cropped_data_in[0],
                                  self.i_conf: cropped_data_in[1],
                                  self.i_gt_D: cropped_data_in[2],
                                  self.i_R: cropped_data_in[3],
                                  self.i_L: cropped_data_in[4],
                                  self.i_gt_invflow: cropped_data_in[5]
                              })   
                idx += 1
                idx = idx % self.dataset_len
            
            else: # no ground truth test
                assert self.idxtype == None
                data_in = self.dataset.get_data(idx)
                cropped_data_in = crop_multiple_imgs(self.height, self.width, 
                                                    self.ori_height, self.ori_width,
                                                    self.crop_type,  10, 10,
                                                    data_in['D_ori'], data_in['conf'],
                                                    data_in['R_ori'],
                                                    data_in['L'],
                                                    )
                self.sess.run(self.enqueue_op,
                              feed_dict={
                                  self.i_D: cropped_data_in[0],
                                  self.i_conf: cropped_data_in[1],
                                  self.i_R: cropped_data_in[2],
                                  self.i_L: cropped_data_in[3],
                              })   
                idx += 1
                idx = idx % self.dataset_len



        
    def _count_param(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        return total_parameters      

            
    def train(self, step=0):
        if self.training_config.is_from_scratch == False:
            self.load_saver_all.restore(self.sess, self.training_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()

        while step <= self.training_config.total_iter + 1:
            _, loss = self.sess.run([self.trainstep, self.loss_optimize])

            if (step == 2 or step == 4 or step == 6) or (step % self.training_config.display_step == 0):
                summary_, = self.sess.run([self.summary_op])
                self.writer.add_summary(summary_, step)

            if step == 0:
                t_start = time.time()

            if step == 50:
                print("50 iterations need time: %4.4f" % (time.time() - t_start))
               
            if step % 50 == 0:
                print("Iter " + str(step) + " loss: " + str(loss))

            if step % self.training_config.snapshot == 0:
                self.model_saver.save(self.sess, self.training_config.save_model_dir, global_step=step)

            step += 1
        
        self.writer.close()
        sys.exit(0)


    def test(self, step=0):
        self.load_saver_all.restore(self.sess, self.testing_config.wlast)
        t = threading.Thread(target=self._fifo)
        t.daemon = True
        t.start()
        out_path = self.testing_config.output_save_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(pjoin(out_path, 'imgs'))

        loss_txt1 = open(pjoin(out_path, 'loss1.txt'), 'w')
        loss_txt2 = open(pjoin(out_path, 'loss2.txt'), 'w')
        tloss1 = 0
        tloss2 = 0
        N = self.dataset_len
        while step < N: 

            loss2, loss1, R, L, fD = self.sess.run([self.refineflow_EPE, self.roughflow_EPE, self.R, self.L, self.filtered_D])
            loss_txt1.write('%f\n' % (loss1))
            loss_txt2.write('%f\n' % (loss2))

            tloss1 += loss1 / N
            tloss2 += loss2 / N
            imsave(pjoin(out_path, 'imgs', ('%s' % step)  + '_R.png'), R[0, :, :, :])
            imsave(pjoin(out_path, 'imgs', ('%s' % step)  + '_L.png'), L[0, :, :, 0])
            imsave(pjoin(out_path, 'imgs', ('%s' % step)  + '_fD.png'), fD[0, :, :, 0])
            step += 1
        print('%d' % (N) + ' testing finished')
        loss_txt1.write('%f\n' % (tloss1))
        loss_txt2.write('%f\n' % (tloss2))
        print(tloss1)
        print(tloss2)
        loss_txt1.close()
        loss_txt2.close()
        sys.exit(0)

    
    def single_test(self, idx):
        print("***************start single test*******************")
        self.load_saver_all.restore(self.sess, self.testing_config.wlast)
        assert self.use_fifo == False and self.gt_flag == False

        out_path = self.testing_config.output_save_dir
        if not os.path.exists(out_path):
            os.makedirs(out_path)
            os.makedirs(pjoin(out_path, 'imgs'))

        data_in = self.dataset.get_data(idx)
        crop_anchor = [0, 0]
        cropped_data_in = crop_multiple_imgs(self.height, self.width, 
                                            self.ori_height, self.ori_width,
                                            self.crop_type,  crop_anchor[0], crop_anchor[1],
                                            data_in['D_ori'], data_in['conf'],
                                            data_in['R_ori'],
                                            data_in['L'],
                                            )
        R, L, fD, D = self.sess.run([self.R, self.L, self.filtered_D, self.D], feed_dict={
                                  self.D: cropped_data_in[0][np.newaxis, ...],
                                  self.conf: cropped_data_in[1][np.newaxis, ...],
                                  self.R: cropped_data_in[2][np.newaxis, ...],
                                  self.L: cropped_data_in[3][np.newaxis, ...],
                                })
        imsave(pjoin(out_path, 'imgs', ('%s' % idx)  + '_R.png'), R[0, :, :, :])
        imsave(pjoin(out_path, 'imgs', ('%s' % idx)  + '_L.png'), L[0, :, :, 0])
        imsave(pjoin(out_path, 'imgs', ('%s' % idx)  + '_fD.png'), fD[0, :, :, 0])
        imsave(pjoin(out_path, 'imgs', ('%s' % idx)  + '_D.png'), D[0, :, :, 0]) 

        sys.exit(0)     








    def train_test_api(self):
        if self.is_training:
            print("***************start training mode*******************")
            model.train()
        else:
            print("***************start testing mode*******************")
            model.test()

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='nogt', help='dataset')
    parser.add_argument('--dataset2_name', type=str, default='real', help='dataset')
    parser.add_argument('--is_training', type=int, default=0, help='True:1 or False:0')
    parser.add_argument('--is_from_scratch', type=int, default=0, help='Pretraining ?')
    parser.add_argument('--pretrain_ckpt_path', type=str, default='none', help='path to weight file')
    parser.add_argument('--save', type=str, default='./Experiment/', help='where to save stuff')
    parser.add_argument('--sigma', type=str, default=0, help='perturbation strength with mvg_aug in testing mode, set 0 to use uniform perturbation, see loaders.py for definition')
    parser.add_argument('--mvg_aug', type=int, default=0, help='set 1 to use multiview geometry augmentation')

    args = parser.parse_args()
    file_path = os.path.abspath(__file__)
    file_dir = os.path.split(file_path)[:-1]
    print("file_dir {}".format(file_dir[0]))
    # module_path = file_dir[0] + '/modules.py'
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    if args.is_training:
        args.save = args.save + 'training_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name
    else:
        args.save = args.save + 'testing_tmp' #+ datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + file_name

    makedirs(args.save)
    os.chdir(args.save)
    logger = get_logger(logpath='log_file',
                    filepath=file_path, package_files=[])
    logger.info(args)

    mvg_aug = True if args.mvg_aug == 1 else False

    model = Model(args.dataset_name, args.dataset2_name, float(args.sigma), args.is_training, mvg_aug=mvg_aug, \
                    is_from_scratch=args.is_from_scratch, pretrain_ckpt_path=args.pretrain_ckpt_path)

    logger.info('Number of parameters: {}'.format(model._count_param()))

    model.train_test_api()    
        


