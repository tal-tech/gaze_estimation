#-*- coding: UTF-8 -*-
from __future__ import division
import onnxruntime as rt
import sys
sys.path.append('.')
assert sys.version_info.major == 3, 'Please upgrade from {:} to Python 3.x'.format(sys.version_info)
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.landmarks_detector.preprocess_sbr import PreProcess_sbr
import numpy as np


class SBR_Inference_Onnx:
    def __init__(self, model_path):
        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()

        self.preprocess = PreProcess_sbr()


    def inference(self, input_image, face):

        h, w, _ = input_image.shape

        image, [a, b, c, d] = self.preprocess.prepare_input(input_image, face)

        inputs_temp = np.expand_dims(np.asarray(image, dtype=np.float32), 0)

        # network forward
        np_batch_landmark, score = self.sess.run([self.output_name[0].name, self.output_name[1].name], {self.input_name: inputs_temp})

        norm_locs = np_batch_landmark[0].transpose(1, 0)

        real_locs = norm_locs
        real_locs[0, :] = (norm_locs[0, :] * a + c + 1.0) / 2.0 * (w - 1)
        real_locs[1, :] = (norm_locs[1, :] * b + d + 1.0) / 2.0 * (h - 1)
        real_locs = real_locs.transpose(1, 0)

        return real_locs, score

