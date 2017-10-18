#!/usr/bin/env python

import argparse
import chainer
from easydict import EasyDict
import fcis
import matplotlib.pyplot as plt
import os
import os.path as osp
import yaml


filepath = osp.abspath(osp.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('-m', '--modelpath', default=None)
    args = parser.parse_args()

    # chainer config for demo
    gpu = args.gpu
    chainer.cuda.get_device_from_id(gpu).use()
    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False

    # load config
    cfgpath = osp.join(filepath, 'cfg', 'demo.yaml')
    with open(cfgpath, 'r') as f:
        config = EasyDict(yaml.load(f))

    target_height = config.target_height
    max_width = config.max_width
    score_thresh = config.score_thresh
    nms_thresh = config.nms_thresh
    mask_merge_thresh = config.mask_merge_thresh
    binary_thresh = config.binary_thresh

    # load label_names
    n_class = len(coco_instance_segmentation_label_names) + 1

    # load model
    model = fcis.models.FCISResNet101(n_class)
    modelpath = args.modelpath
    if modelpath is None:
        modelpath = model.download()
    chainer.serializers.load_npz(modelpath, model)
    model.to_gpu(gpu)

    dataset = fcis.datasets.coco.COCOInstanceSegmentationDataset(
        data_dir=args.data_dir, split='val',
        use_crowded=True, return_crowded=True, return_area=True)
    # load input images
    sizes = list()
    gt_masks = list()
    gt_bboxes = list()
    gt_labels = list()
    gt_crowdeds = list()
    gt_areas = list()
    pred_masks = list()
    pred_bboxes = list()
    pred_labels = list()
    pred_scores = list()
    for i in range(len(dataset)):
        img, gt_mask, gt_bbox, gt_label, gt_crowded, gt_area = dataset[i]
        _, H, W = img.shape
        sizes.append((H, W))
        gt_masks.append(gt_mask)
        gt_bboxes.append(gt_bbox)
        gt_labels.append(gt_label)
        gt_crowdeds.append(gt_crowded)
        gt_areas.append(gt_area)

        # prediction
        # (C, H, W) -> (H, W, C), RGB->BGR
        img = img.transpose((1, 2, 0))[:, :, ::-1]
        outputs = model.predict(
            [img], gpu, target_height, max_width, score_thresh,
            nms_thresh, mask_merge_thresh, binary_thresh)
        pred_masks.append(outputs[0])
        pred_bboxes.append(outputs[1])
        pred_labels.append(outputs[2])

        cls_probs = outputs[3]


        # batch size = 1



if __name__ == '__main__':
    main()
