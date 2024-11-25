#!/bin/bash

# 切换到主目录
cd .. # 返回到主目录的上一级

# 清理已编译的模型文件
COMPILE_DIR="execution_engine/compiled_result" # 定义存放已编译模型的目录
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} + # 删除与 GPT3-175b 相关的编译结果

# 运行 GPT3-7B 使用 8 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 8 --npu_group 1 --npu_mem 100 \ # 使用 8 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n8' --gen > 'evaluation/evaluation5/gpt7b-n8.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n8，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 运行 GPT3-7B 使用 16 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 16 --npu_group 1 --npu_mem 100 \ # 使用 16 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n16' --gen > 'evaluation/evaluation5/gpt7b-n16.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n16，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 运行 GPT3-7B 使用 32 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 1 --npu_mem 100 \ # 使用 32 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n32' --gen > 'evaluation/evaluation5/gpt7b-n32.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n32，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 运行 GPT3-7B 使用 64 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 64 --npu_group 1 --npu_mem 100 \ # 使用 64 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n64' --gen > 'evaluation/evaluation5/gpt7b-n64.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n64，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 运行 GPT3-7B 使用 128 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 128 --npu_group 1 --npu_mem 100 \ # 使用 128 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n128' --gen > 'evaluation/evaluation5/gpt7b-n128.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n128，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 运行 GPT3-7B 使用 256 个 NPU
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 256 --npu_group 1 --npu_mem 100 \ # 使用 256 个 NPU，1 个 NPU 组，每个 NPU 分配 100GB 内存
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation5/gpt7b-n256' --gen > 'evaluation/evaluation5/gpt7b-n256.txt' 2>&1 # 输出目录为 evaluation5/gpt7b-n256，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} + # 删除与 GPT3-6.7b 相关的编译结果

# 类似的注释模式可用于 GPT3-30B 和 GPT3-175B 的评估任务
# 每个任务指定 NPU 数量、分配的内存、数据集路径以及输出日志文件
# 删除编译结果的命令也需要针对每个模型类型进行清理
