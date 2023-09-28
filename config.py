from gaze_estimation.config.config_node import ConfigNode

config = ConfigNode()

# option: MPIIGaze, MPIIFaceGaze
config.mode = 'MPIIFaceGaze'

# transform
config.transform = ConfigNode()
config.transform.mpiifacegaze_face_size = 224
config.transform.mpiifacegaze_gray = False

# Face detector
config.face_detector = ConfigNode()
config.face_detector.mode = 'pc'
config.face_detector.pc = ConfigNode()
config.face_detector.pc.faceDetextor = 'data/pc_landmark_models/faceDetector_PCWIN_V1.0.1.onnx'
config.face_detector.pc.landmarks_106 = 'data/pc_landmark_models/Landmarks_106_PCWIN_V1.0.2.onnx'
config.face_detector.pc.landmarks_need_61index = [0, 2, 5, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 58, 59, 60, 61, 62, 66, 67, 69, 70, 71, 73, 75, 76, 78, 79, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 98]

# Gaze estimator
config.gaze_estimator = ConfigNode()
config.gaze_estimator.mode = 'onnx' # onnx or torch
config.gaze_estimator.torch_checkpoint = 'data/models/mpiifacegaze/resnet_simple_14_56/checkpoint_0015.pth'
config.gaze_estimator.onnx_modelPath = 'data/models/resnet14_224_0008.onnx'
config.gaze_estimator.camera_params = 'data/calib/sample_params.yaml'
config.gaze_estimator.normalized_camera_params = 'data/calib/normalized_camera_params_face.yaml'
config.gaze_estimator.normalized_camera_distance = 1.0


def get_default_config():
    return config.clone()
