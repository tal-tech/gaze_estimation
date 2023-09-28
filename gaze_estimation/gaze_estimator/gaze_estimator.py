from typing import List

import numpy as np
import yacs.config
import cv2
import onnxruntime

from gaze_estimation.gaze_estimator.common import Camera, Face, FacePartsName, MODEL3D
from .head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self, config: yacs.config.CfgNode):
        self._config = config

        self.camera = Camera(config.gaze_estimator.camera_params) # 得到相机参数
        self._normalized_camera = Camera(config.gaze_estimator.normalized_camera_params) # 得到normalized相机参数

        self._landmark_estimator = LandmarkEstimator(config) # 检测人脸特征点
        self._head_pose_normalizer = HeadPoseNormalizer(self.camera, self._normalized_camera, self._config.gaze_estimator.normalized_camera_distance)
        self._gaze_estimation_model = self._load_onnx_model()  # 加载onnx模型


    def _load_onnx_model(self):
        onnx_modelPath = self._config.gaze_estimator.onnx_modelPath
        session = onnxruntime.InferenceSession(onnx_modelPath)

        return session


    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)


    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        MODEL3D.estimate_head_pose(face, self.camera)
        MODEL3D.compute_3d_pose(face)
        MODEL3D.compute_face_eye_centers(face)

        self._head_pose_normalizer.normalize(image, face)
        self._run_mpiifacegaze_onnx_model(face)


    def _run_mpiifacegaze_onnx_model(self, face: Face) -> None:

        input_name = self._gaze_estimation_model.get_inputs()[0].name
        output_name = self._gaze_estimation_model.get_outputs()[0].name

        # preprocess data
        image = face.normalized_image
        size = self._config.transform.mpiifacegaze_face_size
        resize_image = cv2.resize(image, (size, size))

        transpose_img = resize_image.transpose((2, 0, 1))
        transpose_img = transpose_img / 255
        transpose_img = transpose_img.astype(np.float32)
        for i, (t, m, s) in enumerate(zip(transpose_img, [0.406, 0.456, 0.485], [0.225, 0.224, 0.229])):
            transpose_img[i] = (t - m) / s

        input_data = np.expand_dims(transpose_img, axis=0)

        prediction = self._gaze_estimation_model.run([output_name], {input_name: input_data})
        face.normalized_gaze_angles = prediction[0][0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
