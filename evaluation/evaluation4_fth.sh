```bash
#!/bin/bash

# 切换到主目录
cd .. # 返回到主目录的上一级

# 清理已编译的模型文件
COMPILE_DIR="execution_engine/compiled_result" # 定义存放已编译模型的目录
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果

# 运行 GPT3-30B (64,1) 的评估任务（不复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 1 --npu_mem 24 \ # 使用64个NPU，1个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 设置PIM类型为本地，启用子批处理，数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(64,1)-wo-reuse' --gen > 'evaluation/evaluation4/(64,1)-wo-reuse.txt' 2>&1 # 输出目录为 evaluation4/(64,1)-wo-reuse，日志重定向到指定文件

# 运行 GPT3-30B (64,1) 的评估任务（复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 1 --npu_mem 24 \ # 使用64个NPU，1个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(64,1)-w-reuse' --gen > 'evaluation/evaluation4/(64,1)-w-reuse.txt' 2>&1 # 输出目录为 evaluation4/(64,1)-w-reuse，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果

# 运行 GPT3-30B (16,4) 的评估任务（不复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 4 --npu_mem 24 \ # 使用64个NPU，4个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(16,4)-wo-reuse' --gen > 'evaluation/evaluation4/(16,4)-wo-reuse.txt' 2>&1 # 输出目录为 evaluation4/(16,4)-wo-reuse，日志重定向到指定文件

# 运行 GPT3-30B (16,4) 的评估任务（复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 4 --npu_mem 24 \ # 使用64个NPU，4个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(16,4)-w-reuse' --gen > 'evaluation/evaluation4/(16,4)-w-reuse.txt' 2>&1 # 输出目录为 evaluation4/(16,4)-w-reuse，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果

# 运行 GPT3-30B (8,8) 的评估任务（不复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 8 --npu_mem 24 \ # 使用64个NPU，8个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(8,8)-wo-reuse' --gen > 'evaluation/evaluation4/(8,8)-wo-reuse.txt' 2>&1 # 输出目录为 evaluation4/(8,8)-wo-reuse，日志重定向到指定文件

# 运行 GPT3-30B (8,8) 的评估任务（复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 8 --npu_mem 24 \ # 使用64个NPU，8个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(8,8)-w-reuse' --gen > 'evaluation/evaluation4/(8,8)-w-reuse.txt' 2>&1 # 输出目录为 evaluation4/(8,8)-w-reuse，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果

# 运行 GPT3-30B (4,16) 的评估任务（不复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 16 --npu_mem 24 \ # 使用64个NPU，16个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(4,16)-wo-reuse' --gen > 'evaluation/evaluation4/(4,16)-wo-reuse.txt' 2>&1 # 输出目录为 evaluation4/(4,16)-wo-reuse，日志重定向到指定文件

# 运行 GPT3-30B (4,16) 的评估任务（复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 16 --npu_mem 24 \ # 使用64个NPU，16个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(4,16)-w-reuse' --gen > 'evaluation/evaluation4/(4,16)-w-reuse.txt' 2>&1 # 输出目录为 evaluation4/(4,16)-w-reuse，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果

# 运行 GPT3-30B (1,64) 的评估任务（不复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 64 --npu_mem 24 \ # 使用64个NPU，64个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --output 'evaluation/evaluation4/(1,64)-wo-reuse' --gen > 'evaluation/evaluation4/(1,64)-wo-reuse.txt' 2>&1 # 输出目录为 evaluation4/(1,64)-wo-reuse，日志重定向到指定文件

# 运行 GPT3-30B (1,64) 的评估任务（复用）
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 64 --npu_mem 24 \ # 使用64个NPU，64个NPU组，每个NPU分配24GB内存
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \ # 数据集路径为 simulation-time-bs64-seq1024.tsv
    --

output 'evaluation/evaluation4/(1,64)-w-reuse' --gen > 'evaluation/evaluation4/(1,64)-w-reuse.txt' 2>&1 # 输出目录为 evaluation4/(1,64)-w-reuse，日志重定向到指定文件
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} + # 删除与 GPT3-30b 相关的编译结果
```