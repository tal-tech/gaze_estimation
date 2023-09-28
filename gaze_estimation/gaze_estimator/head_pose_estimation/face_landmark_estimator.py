from typing import List

import numpy as np
import yacs.config

from gaze_estimation.gaze_estimator.common import Face

from gaze_estimation.gaze_estimator.pipeline_detect_alignment.inference import Landmarks


class LandmarkEstimator:
    def __init__(self, config: yacs.config.CfgNode):
        self.mode = config.face_detector.mode
        if self.mode == 'pc':
            detector_path = config.face_detector.pc.faceDetextor
            landmarks_path = config.face_detector.pc.landmarks_106
            self.detector = Landmarks(detector_path, landmarks_path)
            self.landmarks_need_61index = config.face_detector.pc.landmarks_need_61index
        else:
            raise ValueError

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        if self.mode == 'pc':
            return self._detect_faces_pc(image)
        else:
            raise ValueError

    def _detect_faces_pc(self, image: np.ndarray) -> List[Face]:

        tmp_img = image.copy()
        count = 0
        last_avg_lamks_center = []
        last_box = []
        last_interocular_distance = 0.
        last_facearea = 0.

        results = self.detector.inference(tmp_img,
                                           count,
                                           last_avg_lamks_center,
                                           last_box,
                                           last_interocular_distance,
                                           last_facearea)
        detected = []
        for result in results:
            box, landmarks, _, _, _, _ = result
            if box is None or landmarks is None:
                continue
            bbox = np.array([int(box[0]), int(box[1]), int(box[2]), int(box[3]), "%0.4f" % box[4]], dtype=np.float)

            landmarks = np.array([list(landmarks[i]) for i in self.landmarks_need_61index], dtype=np.float)
            detected.append(Face(bbox, landmarks))

        return detected
