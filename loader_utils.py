'''
Utility functions for data loading

@Di Qiu, 23-07-2019
'''
import numpy as np


def crop_multiple_imgs(height=384, width=512, 
                        ori_height=480, ori_width=640, 
                        crop_type='random', 
                        x_p_f=0, y_p_f=0,
                        *args):
    if crop_type == 'random':
        x_p_f = np.random.random_integers(0, ori_width - width)
        y_p_f = np.random.random_integers(0, ori_height - height)
    else:
        assert x_p_f < ori_width - width + 1 and y_p_f < ori_height - height + 1 and x_p_f >= 0 and y_p_f >= 0

    out = []
    for img in args:
        nimg = img[y_p_f: y_p_f + height, x_p_f: x_p_f + width, :]
        out.append(nimg)

    return out


def add_newaxis(*args):
    '''
    Add a new axis at the channel dimension if dimension is less than 3
    '''
    output = []
    for arg in args:
        if len(arg.shape) < 3:
            output.append(arg[...,np.newaxis])
        else:
            output.append(arg)
    return output


def delete_singleton_axis(*args):
    '''
    Delete singleton axis 
    '''
    output = []
    for arg in args:
        output.append(np.squeeze(arg))
    return output


def read_flow(filename, h, w):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    data2d = np.fromfile(f, np.float32)
    # print(data2d)
    # reshape data into 3D array (columns, rows, channels)
    data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

