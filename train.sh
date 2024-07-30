#!/bin/bash

# 设置环境变量
export FLAGS_use_cuda_managed_memory=true
export FLAGS_allocator_strategy=auto_growth
export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree

# 获取当前时间
time=$(date +%Y-%m-%d-%H-%M-%S)

# 使用传入的参数作为配置文件路径
config_path=$2

# 使用传入的参数作为 GPU 编号列表
gpus=$1

# 运行 PaddlePaddle 训练命令，并将日志输出到指定文件
python -u -m paddle.distributed.launch --gpus "$gpus" paddlemix/tools/supervised_finetune.py $config_path 2>&1|tee -a log/$time.log