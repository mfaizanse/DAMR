import caffe
import numpy as np
from PIL import Image
import os
import time


class Harmonization:

    def __init__(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        # load net
        self.net = caffe.Net('model/deploy_512.prototxt', 'model/harmonize_iter_200000.caffemodel', caffe.TEST)
        # expected input image size
        self.size = np.array([512, 512])

    def compute(self, mask, image):

        # resize the image and change it to numpy array
        im = image.resize(self.size, Image.BICUBIC)
        im = np.array(im, dtype=np.float32)
        # switch to BGR
        im = im[:, :, ::-1]
        # subtract mean
        mean = im.reshape(-1, im.shape[-1]).mean(0)
        im -= mean
        # make dims C x H x W for Caffe
        im = im.transpose((2, 0, 1))

        # resize the mask and change it to numpy array
        mask = mask.resize(self.size, Image.BICUBIC)
        mask = np.array(mask, dtype=np.float32)

        # subtract mean
        mask -= 128.0
        # add one dimension
        mask = mask[np.newaxis, ...]

        # shape for input (data blob is N x C x H x W), set data
        self.net.blobs['data'].reshape(1, *im.shape)
        self.net.blobs['data'].data[...] = im

        self.net.blobs['mask'].reshape(1, *mask.shape)
        self.net.blobs['mask'].data[...] = mask

        # run net for prediction
        self.net.forward()
        out = self.net.blobs['output-h'].data[0]
        out = out.transpose((1, 2, 0))
        out += mean
        out = out[:, :, ::-1]

        neg_idx = out < 0.0
        out[neg_idx] = 0.0
        pos_idx = out > 255.0
        out[pos_idx] = 255.0

        # save result
        result = out.astype(np.uint8)
        return result


