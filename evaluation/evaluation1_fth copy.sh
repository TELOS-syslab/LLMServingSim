#!/bin/bash 

# 切换到上一级目录
cd .. # 返回到主目录

# 运行 GPT3-7B 的评估任务
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 1 --npu_group 1 --npu_mem 24 \ # 指定模型名称为 GPT3-6.7B，使用1个NPU，1个NPU组，每个NPU分配24GB内存
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n1.json' \ # 数据集路径和GPU网络配置文件
    --output 'evaluation/evaluation1/gpt7b' --fast_run > evaluation/evaluation1/gpt7b.txt 2>&1 # 输出目录为 evaluation1/gpt7b，日志重定向到 gpt7b.txt

# 运行 GPT3-30B 的评估任务
python3 -u main.py --model_name 'gpt3-30b' --npu_num 4 --npu_group 1 --npu_mem 24 \ # 指定模型名称为 GPT3-30B，使用4个NPU，1个NPU组，每个NPU分配24GB内存
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n4.json' \ # 数据集路径和GPU网络配置文件
    --output 'evaluation/evaluation1/gpt30b' --fast_run > evaluation/evaluation1/gpt30b.txt 2>&1 # 输出目录为 evaluation1/gpt30b，日志重定向到 gpt30b.txt

# 运行 LLaMA-7B 的评估任务
python3 -u main.py --model_name 'llama-7b' --npu_num 1 --npu_group 1 --npu_mem 24 \ # 指定模型名称为 LLaMA-7B，使用1个NPU，1个NPU组，每个NPU分配24GB内存
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n1.json' \ # 数据集路径和GPU网络配置文件
    --output 'evaluation/evaluation1/llama7b' --fast_run > evaluation/evaluation1/llama7b.txt 2>&1 # 输出目录为 evaluation1/llama7b，日志重定向到 llama7b.txt

# 运行 LLaMA-30B 的评估任务
python3 -u main.py --model_name 'llama-30b' --npu_num 4 --npu_group 1 --npu_mem 24 \ # 指定模型名称为 LLaMA-30B，使用4个NPU，1个NPU组，每个NPU分配24GB内存
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n4.json' \ # 数据集路径和GPU网络配置文件
    --output 'evaluation/evaluation1/llama30b' --fast_run > evaluation/evaluation1/llama30b.txt 2>&1 # 输出目录为 evaluation1/llama30b，日志重定向到 llama30b.txt
