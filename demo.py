#!/usr/bin/env python
import time
import cv2
from threading import Lock
from utils import load_config, change_x
from gaze_estimation import GazeEstimator
import numpy as np


class Run:

    def __init__(self):
        self.config = load_config()
        self.gaze_estimator = GazeEstimator(self.config)

    def inference(self, frame, screen_width, screen_height):
        """
        :param frame: cv2读入图像，即numpy类型的BGR图像
        :param screen_width: 屏幕宽度，以毫米为单位
        :param screen_height: 屏幕高度，以毫米为单位
        :return: [是否在看屏幕：True/False, 屏幕注视x坐标：单位mm, 屏幕注视y坐标：单位mm]
        """
        try:
            frame = cv2.resize(frame, (640, 480))
        except cv2.error:
            print('Error: empty image')
        else:
            faces = self.gaze_estimator.detect_faces(frame)

            if len(faces) == 0:
                result = [[], []]
            else:
                face = faces[0]

                bbox = face.bbox
                x1, y1, x2, y2, score = bbox
                w = int(y2 - y1)
                h = int(x2 - x1)

                self.gaze_estimator.estimate_gaze(frame, face)
                pre_screen_x, pre_screen_y = self.compute_screen_coordinate(face)

                pre_screen_x += change_x(face)  # 微调x坐标，x轴向左为正，向右为负

                pre_screen_x = float('%0.2f' % pre_screen_x)
                pre_screen_y = float('%0.2f' % pre_screen_y)

                is_screen = False

                if -(screen_width / 2) < pre_screen_x < (screen_width / 2) and 0 < pre_screen_y < screen_height:
                    is_screen = True

                if is_screen:
                    result = [[int(x1), int(y1), w, h, score], [pre_screen_x, pre_screen_y]]
                else:
                    result = [[int(x1), int(y1), w, h, score], []]

            return result

    @staticmethod
    def compute_screen_coordinate(face):
        """三维视线转屏幕二维注视点，适用于摄像头在屏幕中上方"""
        point = face.center
        gaze_vec = face.gaze_vector

        # 米转换成毫米
        X = point[0] * 1000
        Y = point[1] * 1000
        Z = point[2] * 1000

        # 根据射线传播原理计算屏幕注视坐标
        d = -(Z / gaze_vec[2])
        screen_x_mm = X + d * gaze_vec[0]
        screen_y_mm = Y + d * gaze_vec[1]

        face.screen_x = screen_x_mm
        face.screen_y = screen_y_mm

        return screen_x_mm, screen_y_mm

    def estimate_from_picture(self, inputData, screen_width, screen_height):

        if isinstance(inputData, np.ndarray):
            frame = inputData
        else:
            frame = cv2.imread(inputData)

        result = self.inference(frame, screen_width, screen_height)

        return result


st = time.time()
worker = Run()
print("load model cost:", time.time()-st)


def demo():
    st1 = time.time()
    for i in range(10):
        res = worker.estimate_from_picture(inputData=cv2.imread('test_images/test.jpg'), screen_width=525, screen_height=279)
    print(time.time()-st1, res)


g_lock = Lock()


def inference(file_path, width, height):
    with g_lock:
        ret = {
            "face_rectangle": None,
            "sight_spot": None
        }
        rect, spot = worker.estimate_from_picture(file_path, width, height)
        if rect:
            ret['face_rectangle'] = { "x": rect[0], "y": rect[1], "width": rect[2], "height": rect[3] }
        if spot:
            ret['sight_spot'] = {"x": spot[0], "y": spot[1]}
        return ret


if __name__ == '__main__':
    print(inference("test_images/1.jpeg", 525, 279))
	# demo()
