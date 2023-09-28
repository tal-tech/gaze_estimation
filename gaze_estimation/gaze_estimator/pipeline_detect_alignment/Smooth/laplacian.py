
import numpy as np
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.config import PipelineConfig as p_cong

def Laplacian_Smoothing_new(point_now,points,w):
    '''
    :param point_now: 当前landmarks坐标
    :param points:参与平滑的帧所包含的landmrks
    :param w:平滑的超参数，
    :return:
    '''
    weights_int = np.logspace(1,p_cong.SMOOTH_FRAMES,num=p_cong.SMOOTH_FRAMES,base=2)
    weights = weights_int / sum(weights_int)
    L = points-point_now
    L = np.mean([L[i,:,:]*weights[i] for i in range(L.shape[0])],0)
    point_now = point_now+w*L
    return point_now


def Laplacian_Smoothing_new2(points,w):
    '''
    :param point_now: 当前landmarks坐标，shape【2，68/106】（x，y）*点数
    :param points:【n，2，68/106】参与平滑的帧所包含的landmrks，目前n为2，希望可配置
    :param w:平滑的超参数，目前为0.5，希望可配置
    :return:
    '''
    weights_int = np.logspace(2,p_cong.SMOOTH_FRAMES,num=p_cong.SMOOTH_FRAMES,base=2)
    weights = weights_int / sum(weights_int)
    L = points
    point_now = np.mean([L[i]*weights[i] for i in range(len(L))],0)

    return point_now

def Laplacian_Smoothing(point_now,points,w):
    '''
    :param point_now: 当前landmarks坐标，shape【2，68/106】（x，y）*点数
    :param points:【n，2，68/106】参与平滑的帧所包含的landmrks，目前n为2，希望可配置
    :param w:平滑的超参数，目前为0.5，希望可配置
    :return:
    '''
    L = points - point_now
    L = np.mean(L,0)
    point_now = point_now+w*L
    return point_now


def smooth_points(all_landmarks):
    '''
    :param all_landmarks: 所有landmarks，也可以是一段
    :return:
    '''
    new_all_landmarks = []
    for i,landmarks in enumerate(all_landmarks):
        if i > 1 and i<len(all_landmarks)-1:
            points = all_landmarks[i-2:i,...]
            landmarks = Laplacian_Smoothing(landmarks,points,0.5)
        new_all_landmarks.append(landmarks)
    return new_all_landmarks

if __name__ == '__main__':
    landmarks_path = 'result.npy'
    all_landmarks = np.load(landmarks_path)
    smooth_all_landmarks = smooth_points(all_landmarks)
    np.save('smooth_reuslt.npy',smooth_all_landmarks)
