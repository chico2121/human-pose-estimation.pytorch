
import os.path as path
import os
import json
from random import randint
import time

import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from skimage.draw import polygon
import skimage.io as sio
import tracking.tracking_config as tc
from PIL import Image

def showAnns(anns, imgid):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    from matplotlib.collections import PatchCollection

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    np.random.seed(1)
    color_coeffs = np.random.random((31, 3))
    for ann_idx, ann in enumerate(anns):
        if int(ann['image_id']) < int(imgid):
            continue
        if int(ann['image_id']) > int(imgid):
            break
        c_assoc = ann['track_id'] * 97 % 31
        c = (color_coeffs[c_assoc:c_assoc+1, :]*0.6+0.4).tolist()[0]
        if 'keypoints' in ann and type(ann['keypoints']) == list:
            # turn skeleton into zero-based index
            # sks = np.array(coco.loadCats(ann['category_id'])[0]['skeleton'])-1
            kp = np.array(ann['keypoints'])
            x = kp[0::3]
            y = kp[1::3]
            v = kp[2::3]
            # for sk in sks:
            #     if np.all(v[sk]>0):
            #         plt.plot(x[sk],y[sk], linewidth=3, color=c)
            plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
            plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)


def visualize(json_file):
    with open(json_file, 'r') as f:
        file = json.load(f)

    file_pathes = {}
    img_ids = []
    for img_file in file['images']:
        file_path = img_file['file_name']
        file_pathes[img_file['id']] = file_path
    for ann in file['annotations']:
        inst = ann['image_id']
        img_ids.append(inst)
    anns = file["annotations"]

    # for idx, imgid in enumerate(img_ids):
    for idx, imgid in enumerate(file_pathes):
        video_file_name = json_file.split('/')[-1].split('.')[0]
        save_img_path = tc.config.TRACKING.SAVE_IMAGE_PATH
        queue_len_path = os.path.join(save_img_path,"Qlen" + str(tc.config.TRACKING.QUEUE_LEN))
        if not os.path.exists(queue_len_path):
            os.mkdir(queue_len_path)
        dir_name = os.path.join(queue_len_path, video_file_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        vis_folder = os.path.join(dir_name,'vis')
        if not os.path.exists(vis_folder):
            os.mkdir(vis_folder)
        print(idx+1,"/",len(file_pathes))
        file_name = file_pathes[str(imgid)]
        root = tc.config.TRACKING.ROOT
        file_path = os.path.join(root, file_name)
        im = Image.open(file_path)
        width, height = im.size
        fig = plt.figure(figsize=[width*0.01, height*0.01])
        img = sio.imread(file_path)

        # Display.
        plt.clf()
        plt.axis('off')
        plt.imshow(img)

        # Visualize keypoints.
        showAnns(anns, imgid)
        # If you want to save the visualizations somewhere:

        output_file_name = file_name.split("/")[-1].split(".")[0]
        plt.savefig("{}/vis_{}.png".format(vis_folder,output_file_name))
        # Frame updates.
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(1. / 10.)
        # If you want to just look at the first image, uncomment:
        # break
        plt.close()