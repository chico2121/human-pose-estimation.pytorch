# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import json
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import tracking.tracking_init_paths
from tracking.tracking_posetrack_dataset import VIDEODataset
from tracking.tracking_config import config
from tracking.tracking_config import update_config
from tracking.tracking_config import update_dir
from core.loss import JointsMSELoss
from core.function import validate
from nms.nms import oks_iou
from utils.utils import create_logger


import dataset
import models
import json
from collections import deque
from collections import defaultdict
import numpy as np

from detector import human_detection
import tracking.tracking_config as tc
from tracking_pose_estimator import pose_estimator
from tracking.visualization import visualize


def tracking(video_name):

    # args = parse_args()
    # reset_config(config, args)

    tracking_config_file = 'config.yaml'

    detection_config_file = config.TRACKING.DETECTION_CONFIG_PATH
    detection_model_ckp = config.TRACKING.DETECTION_MODEL_PATH
    pose_estimation_config_file = config.TRACKING.POSE_ESTIMATION_CONFIG_PATH
    pose_estimation_model_ckp = config.TRACKING.POSE_ESTIMATION_MODEL_PATH
    root = config.TRACKING.ROOT
    dataset_path = config.TRACKING.DATASET
    video_path = os.path.join(root, dataset_path, video_name)

    update_config(tracking_config_file)

    config.TRACKING.VIDEO_FILE_NAME = video_name
    bbox_thre = config.TEST.IMAGE_THRE
    nms_thre = config.TEST.NMS_THRE
    sim_thre = config.TRACKING.SIM_THRE
    queue_length = config.TRACKING.QUEUE_LEN
    image_name_list = os.listdir(video_path)
    image_name_list.sort()
    image_list=[]
    for idx in range(len(image_name_list)):
        image_list.append(video_path + '/' + image_name_list[idx])

    # image_list = image_list[:20] # for test
    num_frames = len(image_list)

    det_generator = human_detection(detection_config_file, detection_model_ckp, image_list)
    det_result = {} # {'image_name': [candidate bboxes length x [x,y,w,h,score]]}
    for i, img in enumerate(det_generator):
        human_result = img[0]
        instance = []
        for det in human_result:
            if det[4] < bbox_thre:
                continue
            xywh = xyxy2xywh(det)
            instance.append(xywh)
        image_name = image_name_list[i]
        det_result[image_name] = np.array(instance)

    det_result_image_names = [image_list[frame_num].split("/")[-1] for frame_num in range(len(image_list))]
    det_result_first_frame = {}
    det_result_first_frame[det_result_image_names[0]] = det_result[det_result_image_names[0]]


    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(config, is_train=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE is not '':
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        print("please, type the test model file")

    results = pose_estimator(config, model, det_result_first_frame, -1)
    results_first_frame = results[0]

    for idx, person in enumerate(results_first_frame):
        person['track_id'] = idx

    print('first frame have instances: ', len(results_first_frame))

    pose_results = []
    pose_results.append(results_first_frame)

    queue = deque([])
    queue.append(results_first_frame)

    for frame_num in range(num_frames):
        print(str(frame_num+1) + '/' + str(num_frames))


        if frame_num + 1 == num_frames:
            break

        bboxes_flow = flow_box_gen(pose_results[-1])

        key_name = det_result_image_names[frame_num+1]

        if len(det_result[key_name]) == 0:
            print("there is no human in the picture", key_name + "!!!!")
            continue

        bboxes_unified = np.vstack([det_result[key_name], bboxes_flow])
        keep = nms(bboxes_unified, nms_thre)
        nmsed_bboxes = {}
        nmsed_bboxes[key_name] = [bboxes_unified[_keep] for _keep in keep]

        frame_results = pose_estimator(config, model, nmsed_bboxes, frame_num)
        frame_results = frame_results[0]

        sim_matrix = np.array(calculate_similarity_matrix(queue, frame_results))

        instance_length_in_queue = [len(sim_matrix[frame][0]) for frame in range(len(sim_matrix))]

        frame_results = id_alignment(queue, sim_matrix, frame_results, instance_length_in_queue, sim_thre)

        pose_results.append(frame_results)
        queue.append(frame_results)

        if len(queue) > queue_length:
            queue.popleft()

        # print("pose_result_number at frame ", frame_num +1 ,": " ,len(pose_results[frame_num]))

    return pose_results

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
        _bbox[4]
    ]

def id_alignment(queue, sim_matrix, frame_results, instance_length_in_queue, sim_thre):

    track_id = []
    pop_indexes = []

    if len(frame_results) == 0:
        print("frame do not have pose intance!!")
        return frame_results

    for ins_num, person in enumerate(frame_results):
        person_align, track_id_align, argmax_indicator, frame_ind = _id_alignment(track_id, queue, sim_matrix, ins_num, person, instance_length_in_queue, sim_thre)
        if person_align['track_id'] == -1:
            pop_indexes.append(ins_num)
            continue

        frame_results[ins_num] = person_align
        track_id = track_id_align
        # print(frame_results[ins_num]['image'], ins_num, frame_results[ins_num]['track_id'], 'come from frame : ', str(int(queue[frame_ind][argmax_indicator]['image'])-int(frame_results[ins_num]['image'])))

    pop_indexes.sort(reverse=True)
    for index in pop_indexes:
        frame_results.pop(index)

    return frame_results


def _id_alignment(track_id, queue, sim_matrix, ins_num, person, instance_length_in_queue, sim_thre):

    instance_scores = [sim_matrix[i][ins_num] for i in range(len(sim_matrix))]
    concat_ins_scores = np.concatenate(instance_scores)
    argmax_indicator = int(np.argmax(concat_ins_scores))
    frame_ind = 0
    for i, length in enumerate(instance_length_in_queue):
        if argmax_indicator + 1 > length:
            argmax_indicator -= length
            continue
        else:
            frame_ind += i
            break
    id_number = queue[frame_ind][argmax_indicator]['track_id']
    if id_number in track_id:
        if sim_matrix[frame_ind][ins_num][argmax_indicator] < sim_thre:
            person['track_id'] = -1
            return person, track_id, argmax_indicator, frame_ind
        sim_matrix[frame_ind][ins_num][argmax_indicator] = 0
        return _id_alignment(track_id, queue, sim_matrix, ins_num, person, instance_length_in_queue, sim_thre)
    else:
        person['track_id'] = id_number
        track_id.append(id_number)
        return person, track_id, argmax_indicator, frame_ind

def flow_box_gen(frame):

    bboxes = []

    for instance in frame:
        of = optical_flow_loader(config, instance)
        rounded_kpts = np.around(instance['keypoints'][:,:2])
        delta = []
        for kpt in rounded_kpts:
            x = int(kpt[0])
            y = int(kpt[1])
            if x >= of.shape[1] and not y >= of.shape[0] and y >= 0:
                flow = of[y, of.shape[1]-1]
            elif x >= of.shape[1] and not y >= of.shape[0] and y < 0:
                flow = of[0, of.shape[1]-1]
            elif y >= of.shape[0] and not x >= of.shape[1] and x >= 0:
                flow = of[of.shape[0]-1, x]
            elif y >= of.shape[0] and not x >= of.shape[1] and x < 0:
                flow = of[of.shape[0] - 1, 0]
            elif x >= of.shape[1] and y >= of.shape[0]:
                flow = of[of.shape[0]-1, of.shape[1]-1]
            elif x < 0 and not y < 0:
                flow = of[y, 0]
            elif y < 0 and not x < 0:
                flow = of[0, x]
            elif x < 0 and y < 0:
                flow = of[0, 0]
            else:
                flow = of[y, x]
            delta.append(flow)
        kpt_pred = instance['keypoints'][:,:2] + delta

        # kpt_pred = optical_flow_model(config, instance)
        kpt_pred_score = instance['score']
        x,y = np.min(kpt_pred, axis=0)[:2]
        w = np.max(kpt_pred, axis=0)[0] - x
        h = np.max(kpt_pred, axis=0)[1] - y

        x1 = x - 0.075 * w
        y1 = y - 0.075 * h
        x2 = x + 1.075 * w
        y2 = y + 1.075 * h
        w_new = x2-x1
        h_new = y2-y1
        bboxes.append([x1, y1, w_new, h_new, kpt_pred_score])

    return np.array(bboxes)


def optical_flow_model(keypoints):
    # input : num_joints x (corrdinate of keypoints + heatmap)
    # output : keypoint prediction of next frame
    return NotImplementedError


def optical_flow_loader(config, instance):
    # input : keypoint instance
    # output : keypoint prediction of next frame

    root = config.TRACKING.ROOT
    flow_path = config.TRACKING.FLOW_DATASET
    video = config.TRACKING.VIDEO_FILE_NAME
    flow_file = '00' + str(instance['image'])[-4:] + '.flo'

    fn = os.path.join(root, flow_path, video, flow_file)

    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')

        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print('Reading %d x %d flo file\n' % (w, h))
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # print(np.resize(data, (int(h), int(w), 2)).shape)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            # print(np.resize(data, (int(h), int(w), 2)))

    return np.resize(data, (int(h), int(w), 2))


def calculate_similarity_matrix(queue, pose_result):
    # entire_scores : len(queue) x len(pose_result) x len(queue_frame)

    entire_scores = []

    for frame in queue:
        frame_scores = oks_frame(frame, pose_result)
        entire_scores.append(frame_scores)

    return entire_scores

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def oks_frame(queue_frame, current_frame, sigmas=None, in_vis_thre=None):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh, overlap = oks
    :param kpts_db
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    if len(current_frame) == 0:
        return []

    queue_frame_scores = np.array([queue_frame[i]['score'] for i in range(len(queue_frame))])
    queue_frame_kpts = np.array([queue_frame[i]['keypoints'].flatten() for i in range(len(queue_frame))])
    queue_frame_areas = np.array([queue_frame[i]['area'] for i in range(len(queue_frame))])

    current_frame_kpts = np.array([current_frame[i]['keypoints'].flatten() for i in range(len(current_frame))])
    currnet_frame_areas = np.array([current_frame[i]['area'].flatten() for i in range(len(current_frame))])

    oks_ovr_scores = []
    for i in range(len(current_frame)):
        oks_ovr = oks_iou(current_frame_kpts[i], queue_frame_kpts, currnet_frame_areas[i], queue_frame_areas, sigmas, in_vis_thre)
        oks_ovr_scores.append(oks_ovr)

    return oks_ovr_scores

def result2json(pose_result_file, output_file):
    file_data = defaultdict(list)
    video_file = config.TRACKING.VIDEO_FILE_NAME
    dataset_path = config.TRACKING.DATASET
    root = config.TRACKING.ROOT
    video_path = os.path.join(root,dataset_path,video_file)
    image_name_list = os.listdir(video_path)
    image_name_list.sort()
    image_list=[]
    for idx in range(len(image_name_list)):
        # image_list.append(video_path + '/' + image_name_list[idx])
        image_list.append(os.path.join(dataset_path,video_file) + '/' + image_name_list[idx])
    image_id_list = []
    for image_name in image_name_list:
        id = '1'+video_file.split('_')[0]+image_name.split('.')[0][-4:]
        image_id_list.append(id)
    for idx in range(len(image_name_list)):
        images = defaultdict()
        images["file_name"] = image_list[idx]
        images["id"] = image_id_list[idx]
        file_data['images'].append(images)

    for i, img in enumerate(pose_result_file):
        if len(img) == 0:
            print("result do not have result on ", video_file+"/"+str(int(pose_result_file[i]['image_file_name'].split('.')[0])-1)+'.jpg')
            continue
        for j, ins in enumerate(img):
            annotations = defaultdict()
            annotations["image_id"] = ins["image"]
            annotations["track_id"] = ins["track_id"]
            annotations["keypoints"] = list(ins["keypoints"].flatten().astype('float'))
            annotations["scores"] = list(ins["keypoints"][:,2].astype('float'))
            file_data["annotations"].append(annotations)

    categories = []
    cate_dict = defaultdict()
    cate_dict["name"] = "person"
    cate_dict["keypoints"] = \
        ["nose",
         # "upper_neck",
         "heae_bottom",
         "head_top",
         "left_ear",
         "right_ear",
         "left_shoulder",
         "right_shoulder",
         "left_elbow",
         "right_elbow",
         "left_wrist",
         "right_wrist",
         "left_hip",
         "right_hip",
         "left_knee",
         "right_knee",
         "left_ankle",
         "right_ankle"
         ]
    categories.append(cate_dict)

    file_data["categories"] = categories

    with open(output_file, 'w', encoding="utf-8") as make_file:
        json.dump(file_data, make_file, ensure_ascii=False, indent="\t")




if __name__ == '__main__':

    # #for making tracking json files
    # video_name_list = os.listdir(os.path.join(config.TRACKING.ROOT, config.TRACKING.DATASET))
    # for idx, vn in enumerate(video_name_list):
    #     print(vn, " current idx: ",idx+1,"/",len(video_name_list))
    #     video_name = vn
    #     result_path = 'tracking_result'
    #     qlen_name = "Qlen" + str(config.TRACKING.QUEUE_LEN)
    #     if not os.path.exists(os.path.join(result_path,qlen_name)):
    #         os.mkdir(os.path.join(result_path,qlen_name))
    #     result_name = os.path.join(result_path, qlen_name, video_name+'.json')
    #     if os.path.exists(result_name):
    #         continue
    #     results = tracking(video_name)
    #     result2json(results, result_name)


    #for making single tracking json file
    video_name = '003742_mpii_test'
    result_path = 'tracking_result/Qlen20'
    result_name = os.path.join(result_path, video_name+'.json')
    results = tracking(video_name)
    result2json(results, result_name)


    # # for making tracking img files
    # video_name_list = os.listdir(os.path.join(config.TRACKING.ROOT, config.TRACKING.DATASET))
    # for idx, vn in enumerate(video_name_list):
    #     print(vn, " current idx: ",idx+1,"/",len(video_name_list))
    #     video_name = vn
    #     result_path = 'tracking_result'
    #     qlen_name = "Qlen" + str(config.TRACKING.QUEUE_LEN)
    #     if not os.path.exists(os.path.join(result_path,qlen_name)):
    #         os.mkdir(os.path.join(result_path,qlen_name))
    #     result_name = os.path.join(result_path, qlen_name, video_name+'.json')
    #     visualize(result_name)