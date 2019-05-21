import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]

def human_detection(config_file, checkpoint, image_file):
    cfg = mmcv.Config.fromfile(config_file)
    cfg.model.pretrained = None

    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    load_checkpoint(model, checkpoint)

    result = inference_detector(model, image_file, cfg)
    # print(human_result)
    return result

if __name__ == '__main__':
    config = '/root/mmdetection/configs/dcn/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_test_posetrack.py'
    checkpoint = '/root/mmdetection/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x/latest.pth'
    image_file = 'data/posetrack/images/val/000342_mpii_test/000000.jpg'
    human_detection(config, checkpoint, image_file)
