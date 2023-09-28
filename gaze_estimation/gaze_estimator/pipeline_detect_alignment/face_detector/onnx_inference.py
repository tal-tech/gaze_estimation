#-*- coding: UTF-8 -*-

import numpy as np
import onnxruntime as rt
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.face_detector.py_cpu_nms import py_cpu_nms
import cv2
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.config import PipelineConfig as p_cong

class Onnx_inference(object):
    def __init__(self, img_w, img_h, model_path):
        # create runtime session
        self.img_w = img_w
        self.img_h = img_h
        self.sess = rt.InferenceSession(model_path)

        # get output name
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()

    @staticmethod
    def resize_img_keep_ratio(img, target_size):
        old_size = img.shape[0:2]
        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]))
        pad_w = target_size[1] - new_size[1]
        pad_h = target_size[0] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
        return img_new, top, left


    def inference(self, img_raw):

        if type(img_raw) == list:
            input_data = []
            for i in range(len(img_raw)):
                tmp_img_raw = img_raw[i]
                img_h, img_w, _ = tmp_img_raw.shape
                if img_h != self.img_h or img_w != self.img_w:
                    img_resize, top, left = self.resize_img_keep_ratio(tmp_img_raw, (self.img_w, self.img_h))
                else:
                    img_resize = tmp_img_raw
                    top, left = 0, 0

                img = np.float32(img_resize)
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                input_data.append(np.expand_dims(img, 0))

            input_data = np.concatenate(input_data, axis=0)
        else:
            tmp_img_raw = img_raw
            img_h, img_w, _ = tmp_img_raw.shape
            if img_h != self.img_h or img_w != self.img_w:
                img_resize, top, left = self.resize_img_keep_ratio(tmp_img_raw, (self.img_w, self.img_h))

            else:
                img_resize = tmp_img_raw
                top, left = 0, 0
            img = np.float32(img_resize)
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            input_data = np.expand_dims(img, 0)


        # forward model
        tmp_boxes, tmp_prob = self.sess.run([self.output_name[0].name, self.output_name[1].name], {self.input_name: input_data})

        # batch

        dets_list = []
        for i in range(tmp_prob.shape[0]):

            boxes = tmp_boxes[i, :, :]
            prob = tmp_prob[i, :, :]

            boxes[:, :2] = boxes[:, :2] - (boxes[:, 2:] / 2)
            boxes[:, 2:] = boxes[:, 2:] + boxes[:, :2]

            # ignore low scores
            prob = prob[:, 1]

            inds = np.where(prob > 0.8)[0]
            boxes = boxes[inds]
            scores = prob[inds]

            # keep top-K before NMS
            # order = scores.argsort()[::-1][:5000]
            order = scores.argsort()[::-1][:500]
            boxes = boxes[order]
            scores = scores[order]

            # scale to ori img size
            boxes[:, 0::2] *= self.img_w # first to scale to self.img
            boxes[:, 0::2] = boxes[:, 0::2] - left # second 去掉padding部分
            boxes[:, 0::2] /= (self.img_w - left*2) # 归一化
            boxes[:, 0::2] *= img_w # 返回原图大小
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, img_w) # clip borders

            boxes[:, 1::2] *= self.img_h
            boxes[:, 1::2] = boxes[:, 1::2] - top
            boxes[:, 1::2] /= (self.img_h - top*2)
            boxes[:, 1::2] *= img_h
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, img_h)

            # do NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

            keep = py_cpu_nms(dets, 0.4)

            dets = dets[keep, :] # delete score

            dets = dets[:1, ...]

            if dets.any():
                if dets[0][-1] < p_cong.DETECTION_CONF_THRES:
                    dets = np.array([])

            dets_list.append(dets)

        return dets_list[0]
