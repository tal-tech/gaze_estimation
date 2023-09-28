class Config():
    CUDA = False


class PipelineConfig():
    NORMLIZE_DIS_THRES = 5.0
    INTEROCULAR_DIS_NORDIFF = 0.2
    DETECTION_CONF_THRES = 0.9
    # SKIP_FRAMES = 16
    UPDATE_EXCEPTIVE_LANDMARKS = False
    TRUST_THRESHOLD = 0.8
    SMOOTH_FRAMES = 2
    LANDMARKS_QUALITY_THRES = 0.8
    FACE_AREA_RATIO = 0.02  # 0.02
    SMOOTH_W = 0.5
    DOING_SMOOTH = True


class LandmarkConfig():
    pre_crop_expand = 0.2
    crop_width = 112
    crop_height = 112
    model_config = '../configs/Detector.config'
    dataset_name = '300W-68'
    num_pts = 106


class FaceDetectorConfig():
    DETECT_H = 300
    DETECT_W = 300





