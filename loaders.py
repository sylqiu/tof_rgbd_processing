'''
Data loader class implementations for TensorFlow:

-SimulatedToF: synthetic data with optional multiview geometry augmentation;
-RealData: real data with optional multiview geometry augmentation;
-NoGtTest: testing loader without groundtruth loading, optional initial calibration step;

@Di Qiu, 23-07-2019
'''
import collections, os
import numpy as np
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import median_filter, sobel, gaussian_filter
from skimage.color import rgb2gray
import scipy.io as sio
from os.path import join as pjoin
from utils.warp_by_flow import warp_by_flow
from utils.camera_util import *
from loader_utils import *


def get_dataset(dataset_name):
    Dict = {'simtof' : SimulatedToF,
            'real' : RealData,
            'nogt': NoGtTest,
            'none': None
        }
    return Dict[dataset_name]


def plane_correction(fov, img_size, fov_flag=True):
    x, y = np.meshgrid(np.linspace(0, img_size[1]-1, img_size[1]), 
                        np.linspace(0, img_size[0]-1, img_size[0]))
    if fov_flag:
        fov = 63.5 * np.pi / 180
        flen = (img_size[1]/2) / np.tan(fov/2)
    else:
        flen = fov

    x = (x - img_size[1]/2.) / flen
    y = (y - img_size[0]/2.) / flen
    norm = 1. / np.sqrt(x ** 2 + y ** 2 + 1.)
    
    return norm


class RotoTransParam():
    def __init__(self, avgrot=0, avgtrans=30, avgprpt=22):
        self.avgrot = avgrot
        self.avgtrans = avgtrans
        self.avgprpt = avgprpt


class SimulatedToF(object):
    '''
    ToFFlyingThings3D loader
    Max actual depth (4m) corresponds to value 4095. In this loader depth will be normalized to [0,1]
    '''
    def __init__(self, path_to_data, sigma=0,
                 split='train',
                 img_size=(480,640),
                 mvg_aug=True,
                 align_flag=False
                 ):
        self.use_sigma = False
        if sigma > 0:
            self.use_sigma=True
        self.name = 'simtof'
        self.root = os.path.expanduser(path_to_data)
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.mvg_aug = mvg_aug
        self.align = align_flag


        ### set paths ####
        if self.align or self.mvg_aug:
            self.gt_path = 'gt_depth_rgb/'
        else:
            self.gt_path = 'gt_depth_rgb_small_pt/'
        ### end set paths ###
        for split in ['train', 'test']:
            path = pjoin(self.root, split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        
        camparam = param_buffer(path_to_data + 'calib.bin')
        camparam = adjust_rotation(camparam, [1, 0, 0, 0, 1, 0, 0, 0, 1])
        camparam = adjust_translation(camparam, [0, 0, 0])
        camparam = adjust_distorsion(camparam, 0, 0)
        self.camparam = camparam

        rototrans_instance = RotoTransParam()
        self.avgrot = rototrans_instance.avgrot
        self.avgprpt = rototrans_instance.avgprpt
        self.avgtrans = rototrans_instance.avgtrans
        print(img_size)
        self.warp_by_flow = warp_by_flow(img_size[0], img_size[1], 1)

        norm = plane_correction(63.5, img_size)
        # norm = np.expand_dims(norm, 2)
        self.plane_correction = norm
        self.sigma = sigma

        self.len = len(self.files[self.split])

    def __len__(self):
        return self.len


    def _sample_ind(self): 
        return np.random.random_integers(0, self.len - 1)


    def next_random_sample(self):
        idx = self._sample_ind()
        out = self.get_data(idx)
        return out

    def token_list(self):
        '''
        return list of tokens for looping over data
        '''
        return self.files[self.split]

    def get_data(self, idx):

        token = self.files[self.split][idx]

        imgR_ori = imread(self.root + self.gt_path + token + '_rgb.png').astype(
            np.float32) / 255.
        gt_D_data = sio.loadmat(self.root + self.gt_path + token + '_gt_depth.mat')
        gt_D = gt_D_data['gt_depth'].astype(np.float32) * self.plane_correction

        imgL = imread(self.root + 'nToF/' + token + '_noisy_intensity.png').astype(np.float32)
        imgL = np.expand_dims(imgL, 2) / 255.
        imgL = median_filter(imgL, size=5)

        input_D_data = sio.loadmat(self.root + 'nToF/' + token + '_noisy_depth.mat')
        input_D = input_D_data['ndepth'].astype(np.float32)
        D_ori = input_D * self.plane_correction

        conf = np.ones([self.img_size[0], self.img_size[1]])

        if self.mvg_aug:

            if self.use_sigma:
                rcx, rcy = np.random.normal(0, self.sigma, [2])
                rt1, rt2 = np.random.normal(0, self.sigma*2/3, [2])
            else:
                rcx, rcy, rt1, rt2 = np.random.rand(4)
                rcx = self.avgprpt * rcx - self.avgprpt / 2.
                rcy = self.avgprpt * rcy - self.avgprpt / 2.
                rt1 = self.avgtrans * rt1 - self.avgtrans / 2.
                rt2 = self.avgtrans * rt2 - self.avgtrans / 2.


            rt = np.array([rt1, rt2, 0.], np.float32)
            camparam2 = adjust_principal_point(self.camparam, rcx, rcy)
            camparam2 = adjust_translation(camparam2, rt)

            gtwhere = np.less(gt_D, 10).astype(np.float32)
            gt_D = gt_D * (1 - gtwhere) +  gtwhere * 2000
            gt_flow, depthR = compute_gtflow_from_depth(camparam2, gt_D)
            depthR, nconf = self.warp_by_flow(gt_flow, depthR)
            where_dR = np.less(depthR, 10).astype(np.float32)


            conf = conf * (1 - where_dR) * nconf

            depthR = where_dR * 4095 + (1 - where_dR) * depthR
            gt_invflow = compute_inverse_flow_by_depth(camparam2, depthR)

        elif self.align:
            gt_invflow = np.zeros([self.img_size[0], self.img_size[1], 2])

        else:
            gt_invflow = sio.loadmat(self.root + self.gt_path + token + '_gt_invflow.mat')
            gt_invflow = gt_invflow['gt_invflow']

        D_ori = np.expand_dims(D_ori, 2)
        gt_D = np.expand_dims(gt_D, 2)
        conf = np.expand_dims(conf, 2)


        return {
            'L': imgL,
            'R_ori': imgR_ori,
            'D_ori': D_ori / 4095.,
            'conf': conf,
            'gt_invflow': gt_invflow,
            'gt_D': gt_D / 4095.
        }


class RealData(object):
    '''
    Real data loader

    Max actual depth corresponds to value 4095. In this loader depth will be normalized to [0,1]
    '''
    def __init__(self, path_to_data, sigma=0,
                 split='train', 
                 img_size=(480,640),
                 mvg_aug=True,
                 align_flag=False
                 ):
        self.use_sigma = False
        if sigma > 0:
            self.use_sigma=True,
        self.name = 'real'
        self.root = os.path.expanduser(path_to_data + 'RealData/vivo_data/test_vivo5/')
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.mvg_aug = mvg_aug
        self.align = align_flag


        for split in ['train', 'test']:
            path = pjoin(self.root, split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        camparam = param_buffer_st(self.root + 'calib_verify.bin')
        camparam = adjust_rotation(camparam, [1, 0, 0, 0, 1, 0, 0, 0, 1])
        camparam = adjust_translation(camparam, [0, 0, 0])
        camparam = adjust_distorsion(camparam, 0, 0)
        self.path_to_data = path_to_data
        self.camparam = camparam

        rototrans_instance = RotoTransParam()
        self.avgrot = rototrans_instance.avgrot
        self.avgprpt = rototrans_instance.avgprpt
        self.avgtrans = rototrans_instance.avgtrans
        self.warp_by_flow = warp_by_flow(img_size[0], img_size[1], 1)

        flen = 520 #camparam[param_depth]['fx']
        norm = plane_correction(flen, img_size, False)
        # norm = np.expand_dims(norm, 2)
        self.plane_correction = norm
        self.sigma = sigma
        self.len = len(self.files[self.split])
   

    def __len__(self):
        return self.len
    
    def _sample_ind(self):
        return np.random.random_integers(0, self.len - 1)

    def next_random_sample(self):
        idx = self._sample_ind()
        out = self.get_data(idx)
        return out

    def get_data(self, idx):
        
        token = self.files[self.split][idx]
        imgL = imread(self.root + token + '_reg_ir.png')
        imgL = imgL / 255.
        imgL = median_filter(imgL, size=5)
       
        # imgL = np.tile(imgL, [1, 1, 3])

        imgR_ori = imread(self.root + token + '_rgb.png')
        imgR_ori = imgR_ori / 255.

        imgD = np.fromfile(self.root + token + '_reg_depth.raw',
                           dtype=np.int16,
                           sep="") 
        imgD = imgD.reshape(self.img_size).astype(np.float32) * self.plane_correction
        imgD = median_filter(imgD, size=5)
        imgD_where = np.less(imgD, 100).astype(np.float32)
        imgD = imgD_where * 4095. + (1 - imgD_where) * imgD


        # img_conf = rgb2gray(imread(self.root+ token + '_corr_conf.png'))
        # img_conf = img_conf / 255.
        # img_conf[img_conf >0.3] = 1
        img_conf = np.ones([self.img_size[0], self.img_size[1]], dtype=np.float)
        

        if self.mvg_aug:
            if self.use_sigma:
                rcx, rcy = np.random.normal(0, self.sigma, [2]) 
                rt1, rt2 = np.random.normal(0, self.sigma*2./3., [2]) 
            else:
                rcx, rcy, rt1, rt2 = np.random.rand(4)
                rcx = self.avgprpt * rcx - self.avgprpt / 2.
                rcy = self.avgprpt * rcy - self.avgprpt / 2.
                rt1 = self.avgtrans * rt1 - self.avgtrans / 2.
                rt2 = self.avgtrans * rt2 - self.avgtrans / 2.
            


            rt = np.array([rt1, rt2, 0.], np.float32)
            camparam2 = adjust_principal_point(self.camparam, rcx, rcy)
            camparam2 = adjust_translation(camparam2, rt)

            gt_flow, depthR = compute_gtflow_from_depth(camparam2, imgD)

            depthR, nconf = self.warp_by_flow(gt_flow, depthR)

            img_conf = img_conf * nconf

            where_dR = np.less(depthR, 100).astype(np.float32)
            depthR = where_dR * 2000 + (1 - where_dR) * depthR

            img_conf = img_conf * (1 - where_dR)
            gt_invflow = compute_inverse_flow_by_depth(camparam2, depthR)

        elif self.align:
            gt_invflow = np.zeros([self.img_size[0], self.img_size[1], 2])

        else:
            gt_invflow = read_flow(self.root + token + '_flow.flo', self.img_size[0], self.img_size[1])

        imgL = np.expand_dims(imgL, 2)
        imgD = np.expand_dims(imgD, 2)
        img_conf = np.expand_dims(img_conf, 2)


        return {
            'L': imgL,
            'R_ori': imgR_ori,
            'D_ori': imgD / 4095.,
            'gt_invflow': gt_invflow,
            'conf': img_conf,
            'gt_D': imgD / 4095.
        }


class NoGtTest(object):
    '''
    No ground truth loader for deployment

    Assume maximum value 4095. In this loader depth will be normalized to [0,1]
    '''
    def __init__(self,
                 path_to_data, sigma=0,
                 img_size=(480,640), split='test',
                 mvg_aug=False,
                 align_flag=True):

        self.name = 'nogt'
        self.root = os.path.expanduser(path_to_data)
        self.split = 'test'
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.perform_calib = True


        for split in ['test']:
            path = pjoin(self.root, split + '_list.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        self.len = len(self.files[self.split])

        self.warp_by_flow = warp_by_flow(img_size[0], img_size[1], 1)
        
        flen = 520 #camparam[param_depth]['fx']
        norm = plane_correction(flen, img_size, False)
        # norm = np.expand_dims(norm, 2)
        self.plane_correction = norm

    def __len__(self):
        return self.len

    def _sample_ind(self):
        return np.random.random_integers(0, self.len - 1)

    def next_random_sample(self):
        idx = self._sample_ind()
        out = self.get_data(idx)
        return out

    def get_data(self, idx):

        token = self.files[self.split][idx]
        imgR_ori = imread(self.root + token + '_rgb.png').astype(np.float32) / 255.
        imgL = imread(self.root + token + '_reg_ir.png').astype(np.float32)
        imgL = imgL / 255.
        imgL = median_filter(imgL, size=3)

        input_D = np.fromfile(self.root + token + '_reg_depth.raw',
                           dtype=np.int16,
                           sep="")
        input_D = input_D.reshape([480, 640]).astype(np.float32) * self.plane_correction
        conf = np.ones([self.img_size[0], self.img_size[1]], dtype=np.float)
        



        if self.perform_calib:
            camparam = param_buffer_(self.root + 'calib.bin')
            gt_flow, depthR = compute_gtflow_from_depth(camparam, input_D)
            depthR, conf = self.warp_by_flow(gt_flow, depthR)
            imgwL, _ = self.warp_by_flow(gt_flow, imgL)
            in_conf, _ = self.warp_by_flow(gt_flow, conf)
            conf = conf * in_conf
            where_dR = np.less(depthR, 500).astype(np.float32)

            conf = conf * where_dR
            depthR2 = median_filter(depthR/4095, 15)
            depthR = where_dR * depthR2 + (1 - where_dR) * depthR/4095.

            imgwL2 =  median_filter(imgwL, 15)
            imgL = where_dR * imgwL2[:, :] + (1 - where_dR) * imgwL[:, :]
            imgL = np.expand_dims(imgL, 2)
            conf = np.expand_dims(conf, 2)
            input_D = np.expand_dims(depthR, 2)
        else:
            input_D = np.expand_dims(input_D, 2) / 4095.
            imgL = np.expand_dims(imgL, 2)
            conf = np.expand_dims(conf, 2)
        

        return {
            'L': imgL,
            'D_ori': input_D,
            'R_ori': imgR_ori,
            'conf': conf,
        }

