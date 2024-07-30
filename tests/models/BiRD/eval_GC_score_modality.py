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
    # saved processed data name
    save_json_name = args.json_name.replace(".jsonl", "_processed.json")
    for item in items:
        item["processed_response"] = extract_information(item["response"])

    with open(save_json_name, "w") as f:
        json.dump(items, f, indent=2)

    #############
    for item in items:
        ids.append(item["id"])
        annotations.append(item["annotation"])
        responses.append(item["processed_response"])
        images.append(item["image"])
    print(f"Evaluating {save_json_name} ...")

    
    # 定义一个字典，记录每个modality
    modality_score_dict = {'ct': None, 'mr': None, 'x': None, 'pet': None, 'endoscopy': None, 'dermoscopy': None, 'fundus': None, 'ultrasound': None}
    mean_metric_score = {'Bleu_1': 0, 'Bleu_2': 0, 'Bleu_3': 0, 'Bleu_4': 0, 'METEOR': 0, 'ROUGE_L': 0, 'CIDEr': 0, 'SPICE': 0}

    for modality in modality_score_dict.keys():

        ################## tmp_res_file ##################
        results = []
        for image_id, image, caption in zip(ids, images, responses):
            if modality == image.split('_')[0]:
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
        json_name_item = os.path.basename(save_json_name)
        tmp_res_file = f'{os.path.join("/root/infer_res/tmp", json_name_item[:-5])}_tmp_res.json'
        json.dump(results, open(tmp_res_file, 'w'))
        ####################################
        ########### tmp_anno_file ##########
        results = {"annotations": [], "images": []}
        for image_id, image, caption in zip(ids, images, annotations):
            if modality == image.split('_')[0]:
                results["annotations"].append({
                    'image_id': int(image_id),
                    'id': int(image_id),
                    'caption': caption,
                })
                results["images"].append({'id': int(image_id), "image": image})
        tmp_anno_file = f'{os.path.join("/root/infer_res/tmp", json_name_item[:-5])}_tmp_anno.json'
        json.dump(results, open(tmp_anno_file, 'w'))
        #################################


        coco = COCO(tmp_anno_file)
        coco_result = coco.loadRes(tmp_res_file)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.evaluate()
        
        print(modality, coco_eval.eval)
        modality_score_dict[modality] = coco_eval.eval
        for metric in modality_score_dict[modality].keys():
             modality_score_dict[modality][metric] = round(modality_score_dict[modality][metric], 6)
        os.remove(tmp_res_file)
        os.remove(tmp_anno_file)

    print("==========================================")
    for modality in modality_score_dict.keys():
        print(modality.rjust(12),  modality_score_dict[modality])
        for metric in modality_score_dict[modality].keys():
            mean_metric_score[metric] += modality_score_dict[modality][metric]
    

    for metric in mean_metric_score.keys():
        mean_metric_score[metric] /= len(modality_score_dict.keys())
        mean_metric_score[metric] = round(mean_metric_score[metric], 6)
    print("==========================================")
    print('mean socres ', mean_metric_score)

