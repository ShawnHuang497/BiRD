#!/bin/bash
 
# CHECKPOINT_FILE='/root/paddlejob/workspace/env_run/output/paddlemix_med/checkpoints/qwen_vl_sft_ckpts_stage2_240302/checkpoint-final'
# CHECKPOINT_FILE='/root/.paddlenlp/models/qwen-vl/qwen-vl-chat-7b'
CHECKPOINT_FILE='/root/paddlejob/workspace/env_run/output/paddlemix_med/checkpoints/qwen_vl_sft_ckpts_stage2_240318/checkpoint-final'

CUDA_VISIBLE_DEVICES=0 python /root/paddlejob/workspace/env_run/output/paddlemix_med/tests/models/qwen_vl/infer_v3.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/gpt0301/test/0301_CombinedAll_7.5w_test_chatlm_VG_img_coord_OneConv.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish VG Inference."

CUDA_VISIBLE_DEVICES=0 python /root/paddlejob/workspace/env_run/output/paddlemix_med/tests/models/qwen_vl/infer_v3.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/gpt0301/test/0301_CombinedAll_7.5w_test_chatlm_RO_img_coord_OneConv.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish RO Inference."

CUDA_VISIBLE_DEVICES=0 python /root/paddlejob/workspace/env_run/output/paddlemix_med/tests/models/qwen_vl/infer_v3.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/gpt0301/test/0301_CombinedAll_7.5w_test_chatlm_GC_img_coord_OneConv.json \
    --top_p 03 \
    --max_new_tokens 80 &

wait
echo "Finish GC Inference."

CUDA_VISIBLE_DEVICES=0 python /root/paddlejob/workspace/env_run/output/paddlemix_med/tests/models/qwen_vl/infer_v3.py \
    --checkpoint ${CHECKPOINT_FILE} \
    --json_path /root/paddlejob/workspace/env_run/data/huangxiaoshuang/jsons/gpt0301/test/0301_CombinedAll_7.5w_test_chatlm_MII_img_coord_OneConv.json \
    --top_p 03 \
    --max_new_tokens 100 &

wait
echo "Finish MII Inference."