import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import re
import paddle
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from paddlemix import QWenLMHeadModel, QWenVLTokenizer, QwenVLProcessor
from paddlemix.utils.log import logger

def add_abs_img_path_to_question(text, image_path):
    result = f'Picture 1: <img>{image_path}</img>\n{text}'
    return result

class VQADataset(paddle.io.Dataset):
    def __init__(self, test, img_root_path):
        self.test = [json.loads(line) for line in open(test)]
        self.img_root_path = img_root_path

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        data = self.test[idx]

        image_name = data['image']
        image_path = os.path.join(self.img_root_path, image_name)
        if not os.path.exists(image_path):
            print(f"The path '{image_path}' does not exist.")
        question = data['text'][:-8]
        input_text = add_abs_img_path_to_question(question, image_path)
        annotation = data['gpt4_answer']

        return {
            "idx": idx,
            "image_path": image_path,
            "question": input_text,
            "annotation": annotation,
            "input_text": input_text
        }

def collate_fn(batches, tokenizer):
    idxs = [_["idx"] for _ in batches]
    image_paths = [_["image_path"] for _ in batches]
    questions = [_["question"] for _ in batches]
    input_texts = [_["input_text"] for _ in batches]
    annotations = [_["annotation"] for _ in batches]
    input_ids = tokenizer(input_texts, return_tensors="pd", padding="longest")
    return (idxs, input_ids.input_ids, input_ids.attention_mask, questions, annotations, image_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/path/to/checkpoint_directory')
    parser.add_argument('--json_path', type=str, default='/path/to/llava_med_eval_qa50_qa.jsonl')
    parser.add_argument('--top_p', type=str, default='03')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--img_root_path', type=str, default='/path/to/LLaVA_Med_qa50_images/qa50_images')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    random.seed(args.seed)
    paddle.seed(1234)

    dtype = "bfloat16"
    if dtype == "bfloat16" and not paddle.amp.is_bfloat16_supported():
        logger.warning("bfloat16 is not supported on your device, change to float32")
        dtype = "float32"

    model_name = "--".join([args.checkpoint.split('/')[-2], args.checkpoint.split('/')[-1]])
    # model_name = args.checkpoint.split('/')[-1]
    save_json_name = os.path.basename(args.json_path[:-5]) + f"_pred_bfloat16_top{args.top_p}_inferv3_token{args.max_new_tokens}_picture.jsonl"
    # save_json_name = os.path.basename(args.json_path[:-5]) + f"_pred_bfloat16_top{args.top_p}_inferv3_token{args.max_new_tokens}.jsonl"
    save_root = f"/root/infer_res/{model_name}_nosample_nopad"
    # save_root = f"/root/infer_res/{model_name}_nosample_pad"
    os.makedirs(save_root, exist_ok=True)
    res_file = os.path.join(save_root, save_json_name)

    print(f"save_json_name: {res_file}")
    
    # Build tokenizer
    tokenizer = QWenVLTokenizer.from_pretrained(args.checkpoint, dtype=dtype)
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token_id = tokenizer.eod_id

    processor = QwenVLProcessor(tokenizer=tokenizer)

    dataset = VQADataset(test=args.json_path, img_root_path=args.img_root_path)
    dataloader = paddle.io.DataLoader(
        dataset=dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer),
        shuffle=False, num_workers=args.num_workers
    )
        
    # Build model
    model = QWenLMHeadModel.from_pretrained(args.checkpoint, dtype=dtype)
    model.eval()

    all_idx = 0

    for idxs, input_ids, attention_masks, questions, annotations, image_paths in tqdm(dataloader):
        try:
            pred, _ = model.generate(
                                    input_ids=input_ids,
                                    attention_mask=attention_masks,
                                    do_sample=False,
                                    num_beams=1,
                                    max_new_tokens=args.max_new_tokens,
                                    min_new_tokens=1,
                                    length_penalty=1,
                                    num_return_sequences=1,
                                    # output_hidden_states=True,
                                    use_cache=True,
                                    pad_token_id=tokenizer.eod_id,
                                    eos_token_id=tokenizer.eod_id
                                    )   
            responses = [tokenizer.decode(_, skip_special_tokens=False) for _ in pred]

            for i in range(len(idxs)):
                # Combine into a JSON object
                data = {
                    "id": idxs[i],
                    "image": os.path.basename(image_paths[i]),
                    "question": questions[i],
                    "annotation": annotations[i],
                    "response": responses[i]
                }
                # Write JSON object to JSON Lines file
                with open(res_file, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')  # New line for next JSON object
                    all_idx += 1

        except:
            print(questions, image_paths)
