import numpy as np
import cv2
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.config import LandmarkConfig as config


class PreProcess_sbr:
    def __init__(self):
        self.img_width, self.img_height = config.crop_width, config.crop_height
        self.norm_mean = np.array([0.485, 0.456, 0.406])
        self.norm_std = np.array([0.229, 0.224, 0.225])

    def normalize_L(self, x, L):
        return -1. + 2. * x / (L - 1)

    def normalize(self, crop_box, W, H):
        assert len(crop_box) == 4, 'Invalid crop-box : {:}'.format(crop_box)
        x1, y1 = self.normalize_L(crop_box[0], W), self.normalize_L(crop_box[1], H)
        x2, y2 = self.normalize_L(crop_box[2], W), self.normalize_L(crop_box[3], H)
        return x1, y1, x2, y2


    def preprocess(self, img, box):
        ori_img = img.copy()
        # expand box
        h, w, _ = ori_img.shape

        # compute square box
        x1, y1, x2, y2 = box[:4]
        ww = x2 - x1 + 1
        hh = y2 - y1 + 1
        size_w = int(max([ww, hh]) * 0.9)
        size_h = int(max([ww, hh]) * 0.9)
        cx = x1 + ww // 2
        cy = y1 + hh // 2
        x1 = cx - size_w // 2
        x2 = x1 + size_w
        y1 = cy - int(size_h * 0.4)
        y2 = y1 + size_h
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(w, x2))
        y2 = int(min(h, y2))

        ### opencv warp
        scalew = 111. / (x2 - x1)
        scaleh = 111. / (y2 - y1)
        rot_mat = np.float32([[scalew, 0, -1.0 * x1 * scalew], [0, scaleh, -1.0 * y1 * scaleh]])
        resize_img = cv2.warpAffine(img, rot_mat, (112, 112))

        # normalize and compute scale and offset
        x1, y1, x2, y2 = self.normalize((x1, y1, x2, y2), w, h)

        a = (x2 - x1) / 2.0
        b = (y2 - y1) / 2.0
        c = (x2 + x1) / 2.0
        d = (y2 + y1) / 2.0

        resize_img = np.divide(resize_img, 255.)

        mean = np.float64(self.norm_mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.norm_std.reshape(1, -1))

        cv2.subtract(resize_img, mean, resize_img)  # inplace
        cv2.multiply(resize_img, stdinv, resize_img)  # inplace

        return resize_img.transpose(2, 0, 1), [a, b, c, d]

    def prepare_input(self, image, box):
        pre_image, xtheta = self.preprocess(image, box)

        return pre_image, xtheta