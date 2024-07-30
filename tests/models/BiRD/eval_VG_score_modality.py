import argparse
import json
import os
import random
import time
from functools import partial
from typing import Optional
import re
import numpy as np

from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

import numpy as np

from regularization import extract_information

def box_area_np(boxes):
    """
    Compute the area of a set of bounding boxes, given as a 2D numpy array.
    The box is expected in the format [x1, y1, x2, y2].

    Arguments:
    boxes (numpy.ndarray): A 2D numpy array of shape (N, 4) where N is the number of boxes and
    each box is represented as [x1, y1, x2, y2]

    Returns:
    numpy.ndarray: A 1D numpy array of shape (N,) representing the area of each box.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Calculate the Intersection over Union (IOU) between two sets of boxes.
    
    Args:
        boxes1 (np.ndarray): A numpy array containing box1, with shape (N, 4), in the order of top-left and bottom-right.
        boxes2 (np.ndarray): A numpy array containing box2, with shape (M, 4), in the order of top-left and bottom-right.
    
    Returns:
        tuple: Contains numpy arrays for IOU and union, with shapes (N, M) and (N, M) respectively.
    """
    area1 = box_area_np(boxes1)
    area2 = box_area_np(boxes2)

    lt = np.maximum(boxes1[:, np.newaxis, :2], boxes2[:, :2])

    rb = np.minimum(boxes1[:, np.newaxis, 2:], boxes2[:, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def get_float_bboxs(str_bbox_lst):
    """
    Convert a list of string format bounding boxes to a list of float format bounding boxes.
    
    Args:
        str_bbox_lst (List[str]): A list of string format bounding boxes, such as ['(0.5439, 0.0557), (0.7109, 0.2549)'].
    
    Returns:
        np.ndarray: A list of float format bounding boxes with shape (N, 4), where N is the number of boxes, 
        each box represented by four float numbers.
    """
    float_bbox_lst = []
    for str_bbox in str_bbox_lst:
        # Extract numbers from the string
        numbers_str = str_bbox.replace('(', '').replace(')', '')

        # Split the string and convert to float numbers
        float_bbox_lst.append([float(num) for num in numbers_str.split(',')])

    return np.array(float_bbox_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_name", type=str, default="/path/to/test_pred.jsonl")
    args = parser.parse_args()

    ids = []
    annotations = []
    responses = []
    images = []
    items = [json.loads(line) for line in open(args.json_name)]
    print("all items: ", len(items))
    # Save processed data name
    save_json_name = args.json_name.replace(".jsonl", "_processed.json")
    for item in items:
        item["processed_response"] = extract_information(item["response"])
    with open(save_json_name, "w") as f:
        json.dump(items, f, indent=2)
    ##############################
    for item in items:
        ids.append(item["id"])
        annotations.append(item["annotation"])
        responses.append(item["processed_response"])
        images.append(item["image"])
    print(f"Evaluating {args.json_name} ...")

    tp = total_cnt = 0

    PATTERN = r"<box>\(([\d\.,\s]+)\)</box>"
    PATTERN = r"<box>\((.*?)\)</box>"
    PATTERN = r"<box>((.*?))</box>"
    PATTERN = r"<box>(.*?)</box>"

    cnt = 0
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    tp_2 = 0
    fp = fp_2 = 0

    # Dict for each modality
    modality_dict = {'ct': [0,0], 'mr': [0,0], 'x': [0,0], 'pet': [0,0], 'endoscopy': [0,0], 'dermoscopy': [0,0], 'fundus': [0,0], 'ultrasound': [0,0]}

    for image, annotation, response in zip(images, annotations, responses):
        # Get modality
        modality = image.split("_")[0]
        flag = False
        target_bboxs = re.findall(PATTERN, annotation)
        predict_bboxs = re.findall(PATTERN, response)
        # print(target_bboxs, "----->", predict_bboxs)
        if len(predict_bboxs) == 0:
            # For match (587,126),(789,217)
            pattern_2 = r'\(\d+,\d+\),\(\d+,\d+\)'
            predict_bboxs = re.findall(pattern_2, response)
            if predict_bboxs:
                cnt_2 += 1
                flag = True
                # print(target_bboxs, "----->", predict_bboxs)

        # The number of cases where the annotation answer does not have coordinates
        if len(target_bboxs) == 0:
            cnt += 1
            continue
        # The number of cases where the annotation answer exists, but the prediction answer does not have coordinates
        if len(predict_bboxs) == 0:
            cnt_1 += 1
            # print(response)
        else:
            # print(target_bboxs, "----->", predict_bboxs)
            for target_bbox in target_bboxs:
                target_bbox =  [target_bbox]
                for predict_bbox in predict_bboxs:
                    predict_bbox = [predict_bbox]

                    target_bbox_f = get_float_bboxs(target_bbox)
                    try:
                        predict_bbox_f = get_float_bboxs(predict_bbox)
                        iou, _ = box_iou(predict_bbox_f, target_bbox_f)
                    except:
                        print(target_bboxs, "----->", predict_bboxs)
                        iou = [[0]]
                    if iou[0][0] > 0.5:
                        modality_dict[modality][0] += 1
                        tp += 1
                        if flag:
                            tp_2 += 1
                        break
                    else:
                        fp += 1
                        if flag:
                            fp_2 += 1

        modality_dict[modality][1] += len(target_bboxs)
        total_cnt += len(target_bboxs)

print(f'Number of cases where the annotation answer does not have <box></box> coordinates: {cnt}')
print(f'Number of cases where the annotation answer exists, but the prediction answer does not have <box></box> coordinates: {cnt_1}')
print(f'Number of cases where the annotation answer exists, but the prediction answer only has the format (), (): {cnt_2}')
print(f'Number of cases where the annotation answer exists, the prediction answer has the format (), () but is correct: {tp_2}, total correct predictions: {tp}')
print(f'Number of cases where the annotation answer exists, the prediction answer has the format (), () but is incorrect: {fp_2}, total false positives: {fp}')
print(f'Number of false negatives: {total_cnt-tp}')
print(f'Number of <box></box> in the annotation answers: {total_cnt}')

print(modality_dict, '\n')
modality_score = {}
for modality, lst in modality_dict.items():
    modality_score[modality] = lst[0] / (lst[1]+1e-8)
for modality, score in modality_score.items():
    print(f'{modality} recall@0.5: {score}')

all_mean_score = sum(modality_score.values()) / len(modality_score)
print(f'\nall modality mean recall@0.5: {all_mean_score}')
