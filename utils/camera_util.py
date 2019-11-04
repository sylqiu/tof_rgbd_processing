'''
Camera calibration file parsing and  warping functions

Di Qiu, 13-06-2019
'''
import struct
import copy
import tensorflow as tf
import numpy as np

def param_buffer(cam_file):
    param_master = {}
    param_depth = {}
    with open(cam_file) as cam:
        version = struct.unpack('f', cam.read(4))[0]
        cam.seek(124)
        param_master['fx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(156)
        param_master['fy'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(140)
        param_master['cx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(164)
        param_master['cy'] = struct.unpack('d', cam.read(8))[0]
        if version < 3:
            cam.seek(268)
            D0, D1 = struct.unpack('dd', cam.read(16))
            param_master['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]
        else:
            cam.seek(268)
            D0, D1, D2, D3, D4, D5, D6, D7 = struct.unpack('dddddddd', cam.read(8*8))
            param_master['D'] = [D0, D1, D2, D3, D4, D5, D6, D7]
        param_master['height'] = 640
        param_master['width'] = 480

        print('master cx:{}'.format(param_master['cx']))

        cam.seek(196)
        param_depth['fx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(228)
        param_depth['fy'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(212)
        param_depth['cx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(236)
        param_depth['cy'] = struct.unpack('d', cam.read(8))[0]
        if version < 3:
            cam.seek(308)
            D0, D1 = struct.unpack('dd', cam.read(16))
            param_depth['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]
        else:
            cam.seek(380)
            D0, D1, D2, D3, D4, D5, D6, D7 = struct.unpack('dddddddd', cam.read(8 * 8))
            param_depth['D'] = [D0, D1, D2, D3, D4, D5, D6, D7]
        param_depth['height'] = 480
        param_depth['width'] = 640
        cam.seek(28)
        param_depth['R'] = struct.unpack('ddddddddd', cam.read(8 * 9))
        param_master['R'] = param_depth['R']
        cam.seek(100)
        param_depth['t'] = struct.unpack('ddd', cam.read(8*3))
        param_master['t'] = param_depth['t']

        print('depth cx:{}'.format(param_depth['cx']))
    # print(version)
    # print(param_depth['R'])
    #     param_master['fx'] = param_master['fx'] / param_master['width'] * param_depth['width']
    #     param_master['fy'] = param_master['fy'] / param_master['height'] * param_depth['height']
    #     param_master['cx'] = param_master['cx'] / param_master['width'] * param_depth['width'] #+ 11.495109624298727 - 0.9
    #     param_master['cy'] = param_master['cy'] / param_master['height'] * param_depth['height'] #+ 5.084209833794632 + 6.5
        param_master['fx'] = param_depth['fx']
        param_master['fy'] = param_depth['fy']
        param_master['cx'] = param_depth['cx']
        param_master['cy'] = param_depth['cy']
    return {'param_master': param_master, 'param_depth': param_depth}

def param_buffer_st(cam_file):
    param_master = {}
    param_depth = {}
    with open(cam_file) as cam:
        version = struct.unpack('f', cam.read(4))[0]
        cam.seek(124)
        param_master['fx'] = struct.unpack('d', cam.read(8))[0] #* 1.25
        cam.seek(156)
        param_master['fy'] = struct.unpack('d', cam.read(8))[0] #* 1.25

        cam.seek(140)
        param_master['cx'] = struct.unpack('d', cam.read(8))[0] #+ 7.5
        cam.seek(164)
        param_master['cy'] = struct.unpack('d', cam.read(8))[0] #+ 7.5
        if version < 3:
            cam.seek(268)
            D0, D1 = struct.unpack('dd', cam.read(16))
            param_master['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]
        else:
            cam.seek(268)
            D0, D1, D2, D3, D4, D5, D6, D7 = struct.unpack('dddddddd', cam.read(8*8))
            param_master['D'] = [D0, D1, D2, D3, D4, D5, D6, D7]
        param_master['height'] = 3024 # 480
        param_master['width'] = 4032  # 640



        cam.seek(196)
        param_depth['fx'] = struct.unpack('d', cam.read(8))[0] #* 1.25
        cam.seek(228)
        param_depth['fy'] = struct.unpack('d', cam.read(8))[0] #* 1.25
        cam.seek(212)
        param_depth['cx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(236)
        param_depth['cy'] = struct.unpack('d', cam.read(8))[0]
        if version < 3:
            cam.seek(308)
            D0, D1 = struct.unpack('dd', cam.read(16))
            param_depth['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]
        else:
            cam.seek(380)
            D0, D1, D2, D3, D4, D5, D6, D7 = struct.unpack('dddddddd', cam.read(8 * 8))
            param_depth['D'] = [D0, D1, D2, D3, D4, D5, D6, D7]
        param_depth['height'] = 480
        param_depth['width'] = 640
        cam.seek(28)
        param_depth['R'] = struct.unpack('ddddddddd', cam.read(8 * 9))
        param_master['R'] = param_depth['R']
        cam.seek(100)
        param_depth['t'] = np.array(struct.unpack('ddd', cam.read(8*3)), np.float32) * -1 # np.array([-7, -7, 0]) #
        param_master['t'] = param_depth['t'] # np.array([0, 0, 0]) #


    # print(version)
    # print(param_depth['R'])
        param_master['fx'] = param_master['fx'] / param_master['width'] * param_depth['width']
        param_master['fy'] = param_master['fy'] / param_master['height'] * param_depth['height']
        param_master['cx'] = param_master['cx'] / param_master['width'] * param_depth['width']  #+ 11.495109624298727 - 0.9
        param_master['cy'] = param_master['cy'] / param_master['height'] * param_depth['height']  #+ 5.084209833794632 + 6.5
    #     param_master['fx'] = param_depth['fx']
    #     param_master['fy'] = param_depth['fy']
    #     param_master['cx'] = param_depth['cx']
    #     param_master['cy'] = param_depth['cy']
        print('master fx:{}'.format(param_master['fx']))
        print('depth fx:{}'.format(param_depth['fx']))
        print('master cx:{}'.format(param_master['cx']))
        print('depth cx:{}'.format(param_depth['cx']))
    return {'param_master': param_master, 'param_depth': param_depth}


def param_buffer_(cam_file):
    param_master = {}
    param_depth = {}
    with open(cam_file) as cam:
        cam.seek(144)
        param_master['fx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(176)
        param_master['fy'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(160)
        param_master['cx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(184)
        param_master['cy'] = struct.unpack('d', cam.read(8))[0]

        cam.seek(456)
        D0, D1 = struct.unpack('dd', cam.read(16))
        param_master['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]


        param_master['height'] = 480
        param_master['width'] = 640

        cam.seek(216)
        param_depth['fx'] = struct.unpack('d', cam.read(8))[0] 
        cam.seek(248)
        param_depth['fy'] = struct.unpack('d', cam.read(8))[0] 
        cam.seek(232)
        param_depth['cx'] = struct.unpack('d', cam.read(8))[0]
        cam.seek(256)
        param_depth['cy'] = struct.unpack('d', cam.read(8))[0] - 20


        cam.seek(664)
        D0, D1 = struct.unpack('dd', cam.read(16))
        param_depth['D'] = [D0, D1, 0, 0, 0, 0, 0, 0]


        param_depth['height'] = 480
        param_depth['width'] = 640
        cam.seek(72)
        param_depth['R'] = struct.unpack('ddddddddd', cam.read(8 * 9))
        param_master['R'] = param_depth['R']
        cam.seek(0)
        param_depth['t'] = np.array(struct.unpack('ddd', cam.read(8*3)), dtype=np.float32)

        param_master['t'] = param_depth['t']
    # print(version)
    # print(param_depth['R'])
    #     param_master['fx'] = param_master['fx'] / param_master['width'] * param_depth['width']
    #     param_master['fy'] = param_master['fy'] / param_master['height'] * param_depth['height']
    #     param_master['cx'] = param_master['cx'] / param_master['width'] * param_depth['width'] #+ 11.495109624298727 - 0.9
    #     param_master['cy'] = param_master['cy'] / param_master['height'] * param_depth['height'] #+ 5.084209833794632 + 6.5
    #     param_master['fx'] = param_depth['fx']
    #     param_master['fy'] = param_depth['fy']
    #     param_master['cx'] = param_depth['cx']
    #     param_master['cy'] = param_depth['cy']
        print('master fx:{}'.format(param_master['fx']))
        print('depth fx:{}'.format(param_depth['fx']))
        print('master cx:{}'.format(param_master['cx']))
        print('depth cx:{}'.format(param_depth['cx']))
    return {'param_master': param_master, 'param_depth': param_depth}


def adjust_principal_point(camparam, dcx, dcy):
    camparam2 = copy.deepcopy(camparam)
    camparam2['param_master']['cx'] += dcx
    camparam2['param_master']['cy'] += dcy
    return camparam2

def adjust_translation(camparam, t):
    camparam2 = copy.deepcopy(camparam)
    camparam2['param_depth']['t'] = t
    return camparam2

def adjust_rotation(camparam, R):
    camparam2 = copy.deepcopy(camparam)
    camparam2['param_depth']['R'] = R
    return camparam2

def adjust_distorsion(camparam, s1, s2):
    camparam2 = copy.deepcopy(camparam)
    D1 = np.array(camparam2['param_depth']['D'])
    D2 = np.array(camparam2['param_master']['D'])
    camparam2['param_depth']['D'] = D1 * s1
    camparam2['param_master']['D'] = D2 * s2
    return camparam2


def compute_gtflow_from_depth(camparam2, distance):
    camparam = copy.deepcopy(camparam2)
    param_master = camparam['param_master']
    param_depth = camparam['param_depth']
    distance_scale = 1

    t = param_depth['t']
    t = -np.transpose([t])
    # print(t)

    R = param_depth['R']
    R = np.reshape([R], [3, 3])
    R = np.transpose(R)
    t = np.matmul(R, t)
    # print(R)
    fx_master = param_master['fx']
    fy_master = param_master['fy']
    cx_master = param_master['cx']
    cy_master = param_master['cy']
    D_master = param_master['D']

    fx_depth = param_depth['fx']
    fy_depth = param_depth['fy']
    cx_depth = param_depth['cx']
    cy_depth = param_depth['cy']
    D_depth = param_depth['D']

    height = param_depth['height']
    width = param_depth['width']

    x, y = np.meshgrid(range(0, width), range(0, height))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    where = np.less_equal(distance, 50).astype(np.float32)
    where = np.logical_or(where, np.isnan(distance)).astype(np.float32)
    # print(where.max())
    distance = np.nan_to_num(distance) * (1 - where) + 4095 * where
    distance = distance / distance_scale


    p_depth_x = (x - cx_depth) / fx_depth
    p_depth_y = (y - cy_depth) / fy_depth
    p_depth_z = np.ones(np.shape(p_depth_x))
    x2 = p_depth_x
    y2 = p_depth_y

    for i in range(0, 2):
        r2 = x2 * x2 + y2 * y2
        distort_rate = (np.ones(np.shape(p_depth_x)) + D_depth[0] * r2 + D_depth[1] * r2 * r2 +
                        D_depth[4] * r2 * r2 * r2 ) \
                    / (np.ones(np.shape(p_depth_x)) + D_depth[5] * r2 +
                          D_depth[6] * r2 * r2 +
                          D_depth[7] * r2 * r2 * r2)

        x2 = np.divide(p_depth_x, distort_rate) - 2 * D_depth[2] * np.multiply(p_depth_x, p_depth_y) - D_depth[
            3] * (r2 + 2 * np.multiply(p_depth_x, p_depth_x))
        y2 = np.divide(p_depth_y, distort_rate) - 2 * D_depth[3] * np.multiply(p_depth_x, p_depth_y) - D_depth[
            2] * (r2 + 2 * np.multiply(p_depth_y, p_depth_y))
    z = distance / distance_scale * 1. / np.sqrt(x2 * x2 + y2 * y2 + 1)
    p_depth_x = np.multiply(x2, z)
    p_depth_y = np.multiply(y2, z)
    p_depth_z = np.multiply(p_depth_z, z)

    p_depth_x = np.reshape(p_depth_x, [-1])
    p_depth_y = np.reshape(p_depth_y, [-1])
    p_depth_z = np.reshape(p_depth_z, [-1])
    p_depth = np.concatenate([[p_depth_x], [p_depth_y], [p_depth_z]], 0)
    # print('p_depth shape:{}'.format(p_depth.shape))
    # print('t shape:{}'.format(t.shape))
    t = np.tile(t, [1, np.shape(p_depth)[1]])
    # print('t shape:{}'.format(t.shape))
    p_master = np.matmul(R, p_depth) + t

    p_master_x = p_master[0, :]
    p_master_y = p_master[1, :]
    p_master_z = p_master[2, :]

    x1 = np.divide(p_master_x, p_master_z)
    y1 = np.divide(p_master_y, p_master_z)
    r2 = np.multiply(x1, x1) + np.multiply(y1, y1)
    # distort_rate = (1 + D_master[0] * r2 + D_master[1] * r2 * r2 + D_master[4] * r2 * r2 * r2) \
    #                     / (1 + D_master[5] * r2 + D_master[6] * r2 * r2 + D_master[7] * r2 * r2 * r2)
    distort_rate = (np.ones(np.shape(p_depth_x)) + D_master[0] * r2 + D_master[1] * np.multiply(r2, r2) +
                    D_master[4] * np.multiply(r2, np.multiply(r2, r2))) \
                   / (np.ones(np.shape(p_depth_x)) + D_master[5] * r2 +
                      D_master[6] * np.multiply(r2, r2) +
                      D_master[7] * np.multiply(r2, np.multiply(r2, r2)))

    x1 = np.multiply(x1, distort_rate) + 2 * D_master[2] * np.multiply(x1, y1) + D_master[3] * \
         (r2 + 2 * np.multiply(x1, x1))
    y1 = np.multiply(y1, distort_rate) + 2 * D_master[3] * np.multiply(x1, y1) + D_master[2] * \
         (r2 + 2 * np.multiply(y1, y1))

    x1 = x1 * fx_master + cx_master
    y1 = y1 * fy_master + cy_master
    X = np.reshape(x1, np.shape(x))
    Y = np.reshape(y1, np.shape(x))
    Z = np.reshape(p_master_z, np.shape(x))
    # print(X)

    flowx = -x + np.reshape(X, np.shape(x))
    flowy = -y + np.reshape(Y, np.shape(x))
    flow = np.stack([flowx, flowy], 2)
    # print(x)
    # print('flow shape:{}'.format(flow.shape))

    # ZZ = interpolate.interp2d(X, Y, Z, kind='linear', fill_value=0.)
    # ZZ = ZZ(x, y)
    return flow, Z


def compute_inverse_flow_by_depth(camparam2, distance):
    camparam = copy.deepcopy(camparam2)

    param_depth = camparam['param_master']
    param_master = camparam['param_depth']
    distance_scale = 1
    t = param_master['t']
    t = np.transpose([t])
    # print(t)

    R = param_master['R']
    R = np.reshape([R], [3, 3])
    # R = np.transpose(R)
    # t = np.matmul(R, t)

    fx_master = param_master['fx']
    fy_master = param_master['fy']
    cx_master = param_master['cx']
    cy_master = param_master['cy']
    D_master = param_master['D']

    fx_depth = param_depth['fx']
    fy_depth = param_depth['fy']
    cx_depth = param_depth['cx']
    cy_depth = param_depth['cy']
    D_depth = param_depth['D']

    height = param_master['height']
    width = param_master['width']

    x, y = np.meshgrid(range(0, width), range(0, height))
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    where = np.less_equal(distance, 50).astype(np.float32)
    where = np.logical_or(where, np.isnan(distance)).astype(np.float32)
    # print(where.max())
    distance = np.nan_to_num(distance) * (1 - where) + 4095 * where


    p_depth_x = (x - cx_depth) / fx_depth
    p_depth_y = (y - cy_depth) / fy_depth
    p_depth_z = np.ones(np.shape(p_depth_x))
    x2 = p_depth_x
    y2 = p_depth_y

    for i in range(0, 2):
        r2 = x2 * x2 + y2 * y2
        distort_rate = (np.ones(np.shape(p_depth_x)) + D_depth[0] * r2 + D_depth[1] * r2 * r2 +
                        D_depth[4] * r2 * r2 * r2 ) \
                    / (np.ones(np.shape(p_depth_x)) + D_depth[5] * r2 +
                          D_depth[6] * r2 * r2 +
                          D_depth[7] * r2 * r2 * r2)

        x2 = np.divide(p_depth_x, distort_rate) - 2 * D_depth[2] * np.multiply(p_depth_x, p_depth_y) - D_depth[
            3] * (r2 + 2 * np.multiply(p_depth_x, p_depth_x))
        y2 = np.divide(p_depth_y, distort_rate) - 2 * D_depth[3] * np.multiply(p_depth_x, p_depth_y) - D_depth[
            2] * (r2 + 2 * np.multiply(p_depth_y, p_depth_y))
    z = distance / distance_scale * 1. / np.sqrt(x2 * x2 + y2 * y2 + 1)
    p_depth_x = np.multiply(x2, z)
    p_depth_y = np.multiply(y2, z)
    p_depth_z = np.multiply(p_depth_z, z)

    p_depth_x = np.reshape(p_depth_x, [-1])
    p_depth_y = np.reshape(p_depth_y, [-1])
    p_depth_z = np.reshape(p_depth_z, [-1])
    p_depth = np.concatenate([[p_depth_x], [p_depth_y], [p_depth_z]], 0)
    # print('p_depth shape:{}'.format(p_depth.shape))
    # print('t shape:{}'.format(t.shape))
    # print(R)
    t = np.tile(t, [1, np.shape(p_depth)[1]])
    # print('t shape:{}'.format(t.shape))
    p_master = np.matmul(R, p_depth) + t

    p_master_x = p_master[0, :]
    p_master_y = p_master[1, :]
    p_master_z = p_master[2, :]

    x1 = np.divide(p_master_x, p_master_z)
    y1 = np.divide(p_master_y, p_master_z)
    r2 = np.multiply(x1, x1) + np.multiply(y1, y1)
    # distort_rate = (1 + D_master[0] * r2 + D_master[1] * r2 * r2 + D_master[4] * r2 * r2 * r2) \
    #                     / (1 + D_master[5] * r2 + D_master[6] * r2 * r2 + D_master[7] * r2 * r2 * r2)
    distort_rate = (np.ones(np.shape(p_depth_x)) + D_master[0] * r2 + D_master[1] * np.multiply(r2, r2) +
                    D_master[4] * np.multiply(r2, np.multiply(r2, r2))) \
                   / (np.ones(np.shape(p_depth_x)) + D_master[5] * r2 +
                      D_master[6] * np.multiply(r2, r2) +
                      D_master[7] * np.multiply(r2, np.multiply(r2, r2)))

    x1 = np.multiply(x1, distort_rate) + 2 * D_master[2] * np.multiply(x1, y1) + D_master[3] * \
         (r2 + 2 * np.multiply(x1, x1))
    y1 = np.multiply(y1, distort_rate) + 2 * D_master[3] * np.multiply(x1, y1) + D_master[2] * \
         (r2 + 2 * np.multiply(y1, y1))

    x1 = x1 * fx_master + cx_master
    y1 = y1 * fy_master + cy_master
    X = np.reshape(x1, np.shape(x))
    Y = np.reshape(y1, np.shape(x))
    # print(X)

    flowx = -x + np.reshape(X, np.shape(x))
    flowy = -y + np.reshape(Y, np.shape(x))
    flow = np.stack([flowx, flowy], 2)
    # print(x)
    # print('flow shape:{}'.format(flow.shape))
    return flow


def inverse_warp_by_flow(img, flow, h, w):
    _, h_, w_, _ = img.get_shape().as_list()
    gx, gy = tf.meshgrid(range(0, w_), range(0, h_))
    gx = tf.expand_dims(tf.cast(gx, 'float32'), 0)
    gy = tf.expand_dims(tf.cast(gy, 'float32'), 0)
    gx = gx + flow[:, :, :, 0]
    gy = gy + flow[:, :, :, 1]
    wa = bilinear_warp(img, gx, gy)
    wa = tf.image.resize_bilinear(wa, [h, w])
    return wa


def warp_by_flow_cpu(flow, img):
    height = img.shape[0]
    width = img.shape[1]
    x, y = np.meshgrid(range(0, width), range(0, height))
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    X = x + flow[:, :, 0]
    Y = y + flow[:, :, 1]

    warped_img = np.zeros(img.shape)
    occ_img = np.zeros([height, width])
    for idx in range(0, width):
       for idy in range(0, height):
           widx = np.rint(X[idy, idx]).astype(np.int)
           widy = np.rint(Y[idy, idx]).astype(np.int)
           # print(widx)
           # print(widy)
           if widx >= 0 and widx < width and widy >= 0 and widy < height:
               warped_img[widy, widx] = img[idy, idx]
               occ_img[widy, widx] = 1.
    return warped_img, occ_img


def bilinear_warp(img, x, y):

    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]


    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')


    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    # what if no clipping???
    # x0 = tf.clip_by_value(x0, zero, max_x)
    # x1 = tf.clip_by_value(x1, zero, max_x)
    # y0 = tf.clip_by_value(y0, zero, max_y)
    # y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)



    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # print(wa.get_shape)
    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    # print(out.get_shape)
    return out

def get_pixel_value(img, x, y):

    batch_size, height, width, channel_size = img.get_shape().as_list()
    # print('img_size:{}'.format(img.get_shape()))

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, [batch_size, 1, 1, 1])

    channel_idx = tf.range(0, channel_size)
    channel_idx = tf.reshape(channel_idx, [1, 1, 1, channel_size])

    b = tf.tile(batch_idx, [1, height, width, channel_size])
    x = tf.tile(tf.reshape(x, [batch_size, height, width, 1]), [1, 1, 1, channel_size])
    y = tf.tile(tf.reshape(y, [batch_size, height, width, 1]), [1, 1, 1, channel_size])
    c = tf.tile(channel_idx, [batch_size, height, width, 1])

    indices = tf.stack([b, y, x, c], 4)
    # print('ind_shape:{}'.format(indices.get_shape()))

    warped_image = tf.gather_nd(img, indices)
    # print('warped_image_size:{}'.format(warped_image.get_shape()))
    return warped_image
