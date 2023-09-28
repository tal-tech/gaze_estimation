import cv2
import numpy as np
from tqdm import tqdm
import os


class Tracker(object):
    '''
    追踪者模块,用于追踪指定目标
    '''

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        '''
        初始化追踪器种类
        '''
        # 获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN','MOSSE']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器

        if tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()  #pass
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        if tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()

    def initWorking(self, frame, box):
        '''
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
        开启追踪
        '''
        message = None
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                box = [int(self.coord[0]), int(self.coord[1]),
                                      int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])]
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    cv2.rectangle(frame, p1, p2, (0, 0, 255), 2, 1)
            else:
                box = []
        return frame,box


if __name__ == '__main__':
    img_box = np.load('face_box.npy', allow_pickle=True).item()
    img_save_dir = '/Users/tal/Desktop/face/data/beauty_face/KCF_temp'
    img_path_list = list(img_box)
    img_path_list.sort()
    all_time = 0
    track_frame_num = 0
    for i,img_path in tqdm(enumerate(img_path_list)):
        img_name = img_path.split('/')[-1]
        img_save_path = os.path.join(img_save_dir, img_name)

        img = cv2.imread(img_path)
        box = img_box[img_path]

        img = cv2.resize(img,(400,300))
        box = [x/2 for x in box]

        x = int(box[0])
        y = int(box[1])
        w = int(box[2]-box[0])
        h = int(box[3]-box[1])
        box = list(map(int,box))
        box_temp = (x,y,w,h)
        if i % 15 == 0:
            gTracker = Tracker(tracker_type="KCF")
            gTracker.initWorking(img, box_temp)
            cv2.rectangle(img, (box[0],box[1]),(box[2],box[3]), (0, 0, 255), 2, 1)
        else:
            img,tracker_time,box = gTracker.track(img)
            all_time+=tracker_time
            track_frame_num+=1
        cv2.imwrite(img_save_path,img)

    print(all_time/track_frame_num)

