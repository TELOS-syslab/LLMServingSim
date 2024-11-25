#!/bin/bash

# 切换到主目录
cd .. # 返回到主目录的上一级

# 定义已编译模型的目录
COMPILE_DIR="execution_engine/compiled_result" # 已编译结果存放的目录

# 运行 GPT3-7B (4, 1) 的评估任务
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 1 --npu_mem 4 \ # 使用32个NPU，1个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms7B-tp4-pp1.tsv' \ # 设置PIM类型为本地，启用子批处理，数据集路径为 alpaca-bs256-ms7B-tp4-pp1.tsv
    --network 'neupims_(4,1).json' --output 'evaluation/evaluation2/gpt7b-(4,1)' --gen > 'evaluation/evaluation2/gpt7b-(4,1).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件

# 运行 GPT3-7B (2, 2) 的评估任务
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 2 --npu_mem 4 \ # 使用32个NPU，2个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms7B-tp2-pp2.tsv' \ # 数据集路径为 alpaca-bs256-ms7B-tp2-pp2.tsv
    --network 'neupims_(2,2).json' --output 'evaluation/evaluation2/gpt7b-(2,2)' --gen > 'evaluation/evaluation2/gpt7b-(2,2).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的已编译文件

# 运行 GPT3-13B (8, 1) 的评估任务
python3 -u main.py --model_name 'gpt3-13b' --npu_num 64 --npu_group 1 --npu_mem 4 \ # 使用64个NPU，1个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms13B-tp8-pp1.tsv' \ # 数据集路径为 alpaca-bs256-ms13B-tp8-pp1.tsv
    --network 'neupims_(8,1).json' --output 'evaluation/evaluation2/gpt13b-(8,1)' --gen > 'evaluation/evaluation2/gpt13b-(8,1).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件

# 运行 GPT3-13B (4, 2) 的评估任务
python3 -u main.py --model_name 'gpt3-13b' --npu_num 64 --npu_group 2 --npu_mem 4 \ # 使用64个NPU，2个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms13B-tp4-pp2.tsv' \ # 数据集路径为 alpaca-bs256-ms13B-tp4-pp2.tsv
    --network 'neupims_(4,2).json' --output 'evaluation/evaluation2/gpt13b-(4,2)' --gen > 'evaluation/evaluation2/gpt13b-(4,2).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件
find "$COMPILE_DIR" -name "gpt3-13b*" -exec rm -rf {} + # 删除与 GPT3-13b 相关的已编译文件

# 运行 GPT3-30B (8, 2) 的评估任务
python3 -u main.py --model_name 'gpt3-30b' --npu_num 128 --npu_group 2 --npu_mem 4 \ # 使用128个NPU，2个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms30B-tp8-pp2.tsv' \ # 数据集路径为 alpaca-bs256-ms30B-tp8-pp2.tsv
    --network 'neupims_(8,2).json' --output 'evaluation/evaluation2/gpt30b-(8,2)' --gen > 'evaluation/evaluation2/gpt30b-(8,2).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件

# 运行 GPT3-30B (4, 4) 的评估任务
python3 -u main.py --model_name 'gpt3-30b' --npu_num 128 --npu_group 4 --npu_mem 4 \ # 使用128个NPU，4个NPU组，每个NPU分配4GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms30B-tp4-pp4.tsv' \ # 数据集路径为 alpaca-bs256-ms30B-tp4-pp4.tsv
    --network 'neupims_(4,4).json' --output 'evaluation/evaluation2/gpt30b-(4,4)' --gen > 'evaluation/evaluation2/gpt30b-(4,4).txt' 2>&1 # 指定网络配置，输出结果目录和日志文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的已编译文件
