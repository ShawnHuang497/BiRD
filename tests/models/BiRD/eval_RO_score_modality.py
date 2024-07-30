import argparse
import json
import os
import random
import time
from functools import partial
from typing import Optional
 
import paddle
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from paddlemix import QWenLMHeadModel, QwenVLProcessor, QWenVLTokenizer
from paddlemix.utils.log import logger

import re

from regularization import extract_information

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
    ######## saved processed data name ########
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
    print(f"Evaluating {save_json_name} ...")

    fp = tp = tn = fn = total_cnt = 0

    # dict for each modality
    modality_dict = {'ct': [0,0], 'mr': [0,0], 'x': [0,0], 'pet': [0,0], 'endoscopy': [0,0], 'dermoscopy': [0,0], 'fundus': [0,0], 'ultrasound': [0,0]}

    for image, annotation, response in zip(images, annotations, responses):
        modality = image.split("_")[0]
        
        annotation = annotation.lower()
        response = response.lower()

        annotation_tokens = annotation.split(" ")
        flag = True

        for token in annotation_tokens:
            if token not in response:
                flag = False
                
        if flag:
            tp += 1
            modality_dict[modality][0] += 1
            # print(annotation, '--->', response)
        else:
            # print(annotation, '--->', response)
            fp += 1

        total_cnt += 1
        modality_dict[modality][1] += 1


    fn = total_cnt - tp
    recall = tp / (tp+fn+1e-8)
    precision = tp / (tp+fp+1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    acc = (tp) / total_cnt
    print(f"tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}, total_cnt: {total_cnt}")
    print(f"recall: {recall}, precision: {precision}, f1: {f1}, acc: {acc}")


    print(modality_dict, '\n')
    modality_score = {}
    for modality, lst in modality_dict.items():
        modality_score[modality] = lst[0] / (lst[1]+1e-8)
    for modality, score in modality_score.items():
        print(f'{modality} acc@0.5: {score}')

    all_mean_score = sum(modality_score.values()) / len(modality_score)
    print(f'\nall modality mean acc@0.5: {all_mean_score}')