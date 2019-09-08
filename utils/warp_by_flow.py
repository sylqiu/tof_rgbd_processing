import tensorflow as tf
import numpy as np
_warp_by_flow_ops = tf.load_op_library(
    tf.resource_loader.get_path_to_datafile("./ops/build/warp_by_flow.so"))

class warp_by_flow(object):
    def __init__(self, h, w, c):
        self.g2 = tf.Graph()

        with self.g2.as_default():
            self.dx_t = tf.placeholder('float32', [h, w])
            self.dy_t = tf.placeholder('float32', [h, w])
            self.input_img_t = tf.placeholder('float32', [h, w, c])
            self.warped_img_t, self.conf_t = _warp_by_flow_ops.warp_by_flow(self.dx_t, self.dy_t, self.input_img_t)
            self.tmp_sess = tf.Session(graph=self.g2)
    def __call__(self, flow, img):
        dx = np.squeeze(flow[:, :, 0])
        dy = np.squeeze(flow[:, :, 1])
        img = img[:, :, np.newaxis]
        # h, w, c = img.shape




        warped_img, conf = self.tmp_sess.run([self.warped_img_t, self.conf_t],
                                             feed_dict={self.dx_t: dx, self.dy_t: dy, self.input_img_t: img})
        
        warped_img = np.ma.fix_invalid(warped_img, fill_value=0)
        conf = np.ma.fix_invalid(conf, fill_value=0)
        # print(conf.shape)
        # print(warped_img.shape)

        return np.squeeze(warped_img), np.squeeze(conf)

