# coding: utf-8
import cv2
import os
import numpy as np
import sys
sys.path.append('.')
from gaze_estimation.gaze_estimator.pipeline_detect_alignment.config import PipelineConfig as P_cong

class FaceID(object):

    def __init__(self,
                 numID,
                 curRect, landmarks, image):
        self.ID = numID
        self.curRect = curRect
        self.miss_frame = 0
        self.pair = False

        # self.LamksRect_list = [lamks_box]
        # self.ExpRect_list = [expand_box]
        self.Rect_list = [curRect]
        self.landmarks_list = [landmarks]

        # box = self.covert_box(curRect)
        # self.gTracker = Tracker(tracker_type="MEDIANFLOW")
        # self.gTracker.initWorking(image, box)

    def covert_box(self, box):
        x = int(box[0])
        y = int(box[1])
        w = int(box[2] - box[0])
        h = int(box[3] - box[1])
        box_temp = (x, y, w, h)
        return box_temp

    def update(self, newRect, landmarks):
        self.curRect = newRect
        self.landmarks_list.append(landmarks)
        self.Rect_list.append(newRect)
        # self.LamksRect_list.append(lamks_box)
        # self.ExpRect_list.append(expand_box)
        self.pair = True

    def updata_Tracker(self, image, curRect):
        box = self.covert_box(curRect)
        # print('updata--------------------------------------------------------------------')
        # self.gTracker = Tracker(tracker_type="KCF")
        # self.gTracker.initWorking(image, box)

    def get_Tracker_box(self, image):
        img, box = self.gTracker.track(image)
        return box

    def getID(self):
        return self.ID


class IdTracker(object):

    def __init__(self):
        self.IDList = []
        self.IDsignal = 0
        self.trust_threshold = P_cong.TRUST_THRESHOLD
        # self.trust_threshold = P_cong.TRUST_TRACK_LANDMARKS_DIS

    def addID(self,
              curRect, landmarks, image):
        '''
        添加一个人脸
        '''
        tmpID = self.IDsignal
        self.IDList.append(FaceID(tmpID, curRect, landmarks, image))
        # self.IDsignal += 1
        return tmpID


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
        # iou = inter / area_1

        return iou


    def calDIS(self, Lamks_1, Lamks_2):

        # interocular_distance = np.linalg.norm(Lamks_2[66, :2] - Lamks_2[79, :2])
        # dis_sum = sum(np.sqrt(np.sum((Lamks_1[:,:2] - Lamks_2[:,:2]) * (Lamks_1[:,:2] - Lamks_2[:,:2]),axis=-1)))
        # error_dis = dis_sum / (106 * interocular_distance) * 100

        dis_sum = sum(np.sum(np.sqrt((Lamks_1[:,:2] - Lamks_2[:,:2]) * (Lamks_1[:,:2] - Lamks_2[:,:2])),axis=-1))
        error_dis = dis_sum / 106
        return error_dis

    def track(self, curRect, landmarks):
        '''
        找到匹配id并更新landmarks和追踪器
        '''
        if len(self.IDList) == 0:
            return None
        result = [self.calIOU(curRect, tmpID.curRect) for tmpID in self.IDList]
        # result = [self.calDIS(landmarks, tmpID.landmarks_list[-1]) for tmpID in self.IDList]

        tmp_idx = np.argmax(result)

        if result[tmp_idx] >= self.trust_threshold:
            self.IDList[tmp_idx].update(curRect, landmarks)
            return self.IDList[tmp_idx].getID()
        else:
            return None


    def clean_ID(self,):
        if self.IDList:
            self.IDList = []

    # def clean_missing(self, max_miss_frame=0):
    #     if self.IDList:
    #         track_miss_frame = np.asarray([tmpID.miss_frame for tmpID in self.IDList])
    #         del_id = np.nonzero(track_miss_frame >= max_miss_frame)[0]
    #         if del_id is not None:
    #             IDList_temp = self.IDList.copy()
    #             for id in del_id:
    #                 IDList_temp.remove(self.IDList[id])
    #             self.IDList = IDList_temp

            # for tmpID in self.IDList:
            #     if tmpID.pair:
            #         tmpID.miss_frame = 0
            #     else:
            #         tmpID.miss_frame += 1
            #     tmpID.pair = False
            #
            # self.IDsignal = len(self.IDList)
            #
            # # 跟新id
            # if self.IDList:
            #     for i in range(len(self.IDList)):
            #         self.IDList[i].ID = i



if __name__ == "__main__":
    img_box = np.load('face_box.npy', allow_pickle=True).item()
    img_save_dir = '/Users/tal/Desktop/face/data/beauty_face/box'
    for img_path in img_box:
        box = img_box[img_path]
        box = list(map(int, box))
        img = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        img_save_path = os.path.join(img_save_dir, img_name)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        cv2.imwrite(img_save_path, img)


