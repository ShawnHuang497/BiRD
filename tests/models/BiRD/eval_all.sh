#!/bin/bash
 
JSON_FILE='/root/infer_res/qwen_vl_sft_ckpts_stage2_240303--checkpoint-final_nosample_pad'


python /root/paddlejob/workspace/env_run/PaddleMIX/tests/models/qwen_vl/eval_MII_score_modality.py --json_name ${JSON_FILE}/0301_CombinedAll_7.5w_test_chatlm_MII_img_coord_pred_bfloat16_top03_inferv3.jsonl &
wait
echo "Finish MII Inference."

python /root/paddlejob/workspace/env_run/PaddleMIX/tests/models/qwen_vl/eval_GC_score_modality.py --json_name ${JSON_FILE}/0301_CombinedAll_7.5w_test_chatlm_GC_img_coord_pred_bfloat16_top03_inferv3.jsonl &
wait
echo "Finish GC Inference."

python /root/paddlejob/workspace/env_run/PaddleMIX/tests/models/qwen_vl/eval_VG_score_modality.py --json_name ${JSON_FILE}/0301_CombinedAll_7.5w_test_chatlm_VG_img_coord_pred_bfloat16_top03_inferv3.jsonl &
wait
echo "Finish VG Inference."

python /root/paddlejob/workspace/env_run/PaddleMIX/tests/models/qwen_vl/eval_RO_score_modality.py --json_name ${JSON_FILE}/0301_CombinedAll_7.5w_test_chatlm_RO_img_coord_pred_bfloat16_top03_inferv3.jsonl &
wait
echo "Finish RO Inference."
