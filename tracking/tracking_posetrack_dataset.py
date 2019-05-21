# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
import copy
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.cocoeval import COCOeval

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms

logger = logging.getLogger(__name__)


class VIDEODataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, bbox_result, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.joints_thre = cfg.TEST.JOINTS_THRE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.is_train = is_train

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self.parent_ids = None

        self.bbox_result = bbox_result

        self.db = self._get_db()

        # logger.info('=> load {} samples'.format(len(self.db)))


    def _get_db(self):
        gt_db = self._load_person_detection_results()
        return gt_db

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def _load_person_detection_results(self):
        all_imgs = self.bbox_result

        # logger.info('=> Total imgs: {}'.format(len(all_imgs)))

        kpt_db = []
        img_path = os.path.join(self.root, self.image_set)
        for img_name, det_results in all_imgs.items():
            imgnum = img_name.split('.')[0]
            for det_res in det_results:
                box = det_res[:4]
                score = det_res[4]

                if score < self.image_thre:
                    continue

                center, scale = self._box2cs(box)
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
                joints_3d_vis = np.ones(
                    (self.num_joints, 3), dtype=np.float)
                kpt_db.append({
                    'image': os.path.join(img_path, img_name),
                    'filename': img_name,
                    'imgnum': imgnum,
                    'center': center,
                    'scale': scale,
                    'score': score,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                })

        return kpt_db

    # need double check this API and classes field
    def evaluate(self, preds, all_boxes, img_path):

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            split_img_path = img_path[idx].split("/")
            image_id = '1' + split_img_path[-2][:6] + split_img_path[-1][2:6]
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(image_id),
                # 'image_path': img_path,
                'image_file_name': split_img_path[-1]
                # 'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        joints_thre = self.joints_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            after_drop_kpts = []
            for i, n_p in enumerate(img_kpts):
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
                n_p['kpt_score'] = kpt_score

                if kpt_score >= joints_thre:
                    after_drop_kpts.append(n_p)

            if len(after_drop_kpts) == 0:
                print("all kpts' score are smaller than joints threshold!!")
                # print([img_kpts[i]['kpt_score'] for i in range(len(img_kpts))])
                keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
                # print('keep: ', keep)
                if len(keep) == 0:
                    oks_nmsed_kpts.append(img_kpts)
                else:
                    oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])
            else:
                # print([after_drop_kpts[i]['kpt_score'] for i in range(len(after_drop_kpts))])
                keep = oks_nms([after_drop_kpts[i] for i in range(len(after_drop_kpts))], oks_thre)
                # print('keep: ', keep)
                if len(keep) == 0:
                    oks_nmsed_kpts.append(after_drop_kpts)
                else:
                    oks_nmsed_kpts.append([after_drop_kpts[_keep] for _keep in keep])

            # keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)
            #
            # if len(keep) == 0:
            #     oks_nmsed_kpts.append(img_kpts)
            # else:
            #     oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        return oks_nmsed_kpts


