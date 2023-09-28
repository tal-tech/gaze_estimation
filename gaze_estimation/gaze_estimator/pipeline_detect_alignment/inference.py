#-*- coding: UTF-8 -*-
import cv2
import numpy as np
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.landmarks_detector.onnx_inference import SBR_Inference_Onnx
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.face_detector.onnx_inference import Onnx_inference
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.config import FaceDetectorConfig as d_cong, PipelineConfig as p_cong
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.Smooth.tracker import IdTracker
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.Smooth.laplacian import Laplacian_Smoothing


class Landmarks:
    def __init__(self, detection_model_path, landmarks_model_path):
        model_type = landmarks_model_path.split('.')[-1]
        if model_type == 'onnx':
            self.SBR = SBR_Inference_Onnx(landmarks_model_path)
        model_type = detection_model_path.split('.')[-1]
        if model_type == 'onnx':
            self.detector = Onnx_inference(d_cong.DETECT_W, d_cong.DETECT_H, detection_model_path)
        self.normlize_dis_thres = p_cong.NORMLIZE_DIS_THRES
        self.interocular_dis_nordiff = p_cong.INTEROCULAR_DIS_NORDIFF
        # self.skip_frames = p_cong.SKIP_FRAMES
        self.smooth_frames = p_cong.SMOOTH_FRAMES
        self.update_Exceptive_landmarks = p_cong.UPDATE_EXCEPTIVE_LANDMARKS
        self.doing_smooth = p_cong.DOING_SMOOTH
        self.lamks_quality_thres = p_cong.LANDMARKS_QUALITY_THRES
        self.face_area_ratio = p_cong.FACE_AREA_RATIO

        self.smooth_w = p_cong.SMOOTH_W
        self.idTracker = IdTracker()

    def calIOU(self, Rect_1, Rect_2):
        x1_1, y1_1, x1_2, y1_2 = Rect_1
        x2_1, y2_1, x2_2, y2_2 = Rect_2

        area_1 = (x1_2 - x1_1) * (y1_2 - y1_1)
        area_2 = (x2_2 - x2_1) * (y2_2 - y2_1)
        xx1 = max(x1_1, x2_1)
        yy1 = max(y1_1, y2_1)
        xx2 = min(x1_2, x2_2)
        yy2 = min(y1_2, y2_2)
        w = max(0.0, xx2 - xx1)
        h = max(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (area_1 + area_2 - inter)
        return iou

    def compute_dis(self, point_1, point_2, landmarks):
        interocular_distance = np.linalg.norm(landmarks[66, :] - landmarks[79, :])
        offisite_x = point_2[0] - point_1[0]
        offisite_y = point_2[1] - point_1[1]
        normlize_dis = np.sqrt(offisite_x ** 2 + offisite_y ** 2) / interocular_distance * 100
        face_area = (np.max(landmarks[:, 0])-np.min(landmarks[:, 0])) * (np.max(landmarks[:, 1])-np.min(landmarks[:, 1]))

        return [offisite_x, offisite_y], normlize_dis, interocular_distance, face_area


    def inference(self, frame, frame_count, last_avg_lamks_center, last_box, last_interocular_distance, last_facearea):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        result = []

        # if frame_count % self.skip_frames == 0 or last_box is None:
        if frame_count == 0 or last_box is None:
            if self.doing_smooth:
                self.idTracker.clean_ID()
            boxes = self.detector.inference(frame) # 检测人脸
            if boxes.shape[0] != 0:
                box = boxes[0]
            else:
                # 如果检测器没有检测到人脸，就输出None，提前返回
                result.append([None, None, None, None, None, None])
                return result
            last_box, last_interocular_distance, last_facearea = None, None, None
            last_avg_lamks_center = None
            last_landmarks = None
        else:
            # 如果不是第一帧图像，先初始化检测框为上一帧的框
            box = last_box

        landmarks, score = self.SBR.inference(image, box) # 检测landmarks

        ### 判断landmarks的输出score, 如果低于阈值就不输出
        if score <= self.lamks_quality_thres:
            result.append([None, None, None, None, None, None])
            return result

        # 获得lamks_center以及外眼角距离以及face area
        avg_x = np.mean(landmarks[:, 0])
        avg_y = np.mean(landmarks[:, 1])

        avg_lamks_center = [avg_x, avg_y]
        interocular_distance = np.linalg.norm(landmarks[66, :] - landmarks[79, :])
        face_area = (np.max(landmarks[:, 0])-np.min(landmarks[:, 0])) * (np.max(landmarks[:, 1])-np.min(landmarks[:, 1]))

        ###############
        # 如果当前帧数>1, 执行此处。
        # if frame_count > 0  and frame_count % self.skip_frames !=0 and last_avg_lamks_center is not None:
        if frame_count > 0 and last_avg_lamks_center is not None:

            # 计算当前lamks_center与上一帧lamks_center的距离以及外眼角距离
            lamks_center_dis, normlize_dis, interocular_distance, face_area = self.compute_dis(last_avg_lamks_center, avg_lamks_center, landmarks)

            # 对框偏移，并计算外眼角距离的绝对差值（当前帧与上一帧）
            new_box = box.copy()
            box[0::2] = new_box[0::2] + lamks_center_dis[0]
            box[1::2] = new_box[1::2] + lamks_center_dis[1]

            interocular_dis_nordiff = np.abs(interocular_distance - last_interocular_distance) / last_interocular_distance # 外眼角距离不能超过阈值（处理脸前后动的情况）
            face_area_ratio = np.abs((last_facearea / face_area) - 1) ## 计算上一帧人脸size与当前帧人脸size的比值与1的差值

            # # , 如果小于阈值，跳过此处继续执行之后的code, 如果大于阈值，使用人脸检测器进行检测
            if np.abs(normlize_dis) >= self.normlize_dis_thres or interocular_dis_nordiff >= self.interocular_dis_nordiff or face_area_ratio >= self.face_area_ratio:

                ##################
                boxes = self.detector.inference(frame) # 直接重新检测整张图
                if boxes.shape[0] != 0:
                    box = boxes[0]
                else:
                    result.append([None, None, None, None, None, None])
                    return result

                ############ 再次对偏移较大的框校对关键点
                if self.update_Exceptive_landmarks:
                    landmarks, score = self.SBR.inference(image, box)

                    ### 判断landmarks的输出score, 如果低于阈值就不输出
                    if score <= self.lamks_quality_thres:
                        result.append([None, None, None, None, None, None])
                        return result

                    # 获得lamks_center以及外眼角距离
                    avg_x = np.mean(landmarks[:, 0])
                    avg_y = np.mean(landmarks[:, 1])
                    avg_lamks_center = [avg_x, avg_y]
                    interocular_distance = np.linalg.norm(landmarks[66, :] - landmarks[79, :])
                    face_area = (np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])) * (
                                np.max(landmarks[:, 1]) - np.min(landmarks[:, 1]))

                if self.doing_smooth:
                    self.idTracker.clean_ID()

                ### doing smooth
                if self.doing_smooth:
                    # get faceID and updata landmarks （通过iou>0.8）
                    box_id = self.idTracker.track(box, landmarks)
                    # init track
                    if box_id is None:
                        box_id = self.idTracker.addID(box, landmarks, frame)

                    else:
                        # # # Smoothing
                        landmarks_before = self.idTracker.IDList[box_id].landmarks_list

                        if len(landmarks_before) > self.smooth_frames:
                            landmarks_before = landmarks_before[-self.smooth_frames-1:-1]
                            landmarks = Laplacian_Smoothing(landmarks, landmarks_before, self.smooth_w)


        result.append([box, landmarks, avg_lamks_center, score, interocular_distance, face_area])

        return result

