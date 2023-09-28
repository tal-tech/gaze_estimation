import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gaze_estimation.gaze_estimator.common import (Camera, FaceParts, FacePartsName)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera, normalized_distance: float):
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance

    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot) # 根据参考点（鼻尖或眼睛的3D位置）得到人脸3D head pose
        self._normalize_image(image, eye_or_face)
        self._normalize_head_pose(eye_or_face)

    def _normalize_image(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix) # 相机内参求逆
        normalized_camera_matrix = self.normalized_camera.camera_matrix # 标准化的相机内参

        scale = self._get_scale_matrix(eye_or_face.distance) # 缩放矩阵
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix() # M=SR

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv # W=Cs * M * Cr-1

        normalized_image = cv2.warpPerspective(image, projection_matrix, (self.normalized_camera.width, self.normalized_camera.height))
        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image

    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        z_axis = _normalize_vector(center.ravel())
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot[:, 0] # x方向的旋转向量
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ],
                        dtype=np.float)
