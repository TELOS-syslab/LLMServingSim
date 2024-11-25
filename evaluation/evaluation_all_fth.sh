#!/bin/bash

# 开始执行脚本的时间
echo "Start Evaluation" # 输出“开始评估”

# 定义一个函数，将时间格式化为小时、分钟和秒
format_time() {
  local total_seconds=$1 # 获取总秒数
  local hours=$((total_seconds / 3600)) # 计算小时数
  local minutes=$(((total_seconds % 3600) / 60)) # 计算分钟数
  local seconds=$((total_seconds % 60)) # 计算剩余秒数
  echo "${hours}h ${minutes}m ${seconds}s" # 返回格式化的时间字符串
}

# 记录脚本开始时间
start_script=$SECONDS # 获取当前时间（以秒为单位）

# 执行评估1
start=$SECONDS # 记录评估1的开始时间
./evaluation1.sh # 执行评估1脚本
end=$SECONDS # 记录评估1的结束时间
elapsed=$((end - start)) # 计算评估1耗时
echo "Evaluation 1 took $(format_time $elapsed)." # 输出评估1耗时

# 执行评估2
start=$SECONDS # 记录评估2的开始时间
./evaluation2.sh # 执行评估2脚本
end=$SECONDS # 记录评估2的结束时间
elapsed=$((end - start)) # 计算评估2耗时
echo "Evaluation 2 took $(format_time $elapsed)." # 输出评估2耗时

# 执行评估3
start=$SECONDS # 记录评估3的开始时间
./evaluation3.sh # 执行评估3脚本
end=$SECONDS # 记录评估3的结束时间
elapsed=$((end - start)) # 计算评估3耗时
echo "Evaluation 3 took $(format_time $elapsed)." # 输出评估3耗时

# 执行评估4
start=$SECONDS # 记录评估4的开始时间
./evaluation4.sh # 执行评估4脚本
end=$SECONDS # 记录评估4的结束时间
elapsed=$((end - start)) # 计算评估4耗时
echo "Evaluation 4 took $(format_time $elapsed)." # 输出评估4耗时

# 执行评估5
start=$SECONDS # 记录评估5的开始时间
./evaluation5.sh # 执行评估5脚本
end=$SECONDS # 记录评估5的结束时间
elapsed=$((end - start)) # 计算评估5耗时
echo "Evaluation 5 took $(format_time $elapsed)." # 输出评估5耗时

# 记录脚本结束时间
end_script=$SECONDS # 获取当前时间（以秒为单位）
total_elapsed=$((end_script - start_script)) # 计算脚本总耗时
echo "Total time taken: $(format_time $total_elapsed)." # 输出脚本总耗时
