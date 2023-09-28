# -*- coding: utf-8 -*-

import numpy as np
import argparse
import yacs.config

from config import get_default_config


def load_config() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    # args = parser.parse_args()

    config = get_default_config()
    # if args.config is not None:
    #     config.merge_from_file(args.config)
    # config.merge_from_list(args.options)
    config.freeze()
    return config


def dist_l2(real, pred):
    """计算l2距离"""
    real_vec = np.array(real)
    pred_vec = np.array(pred)

    dist = np.linalg.norm(real_vec - pred_vec)

    return dist


def change_x(face):
    """
    微调x坐标
    """

    if face.screen_x < -98:
        return 20.

    return 0.
