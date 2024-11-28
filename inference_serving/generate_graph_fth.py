import os  # 导入 os 模块，用于与操作系统交互，如文件路径操作
import subprocess  # 导入 subprocess 模块，用于执行外部命令
from time import time  # 从 time 模块导入 time 函数，用于计算时间
from .request import *  # 导入当前模块下的 request 模块中的所有内容

def generateGraph(batch, parallel, npu_group, npu_num):  
    # 打印日志（可以注释掉）
    # print("generateGraph start")  
    cwd = os.getcwd()  # 获取当前工作目录
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")  # 设置 chakra 目录路径
    os.chdir(chakra)  # 切换到 chakra 目录

    model = batch.model  # 获取 batch 对象的模型名称
    batch_size = batch.batch_size  # 获取 batch 对象的 batch size
    input_len = batch.input  # 获取 batch 对象的输入长度
    output_len = batch.output  # 获取 batch 对象的输出长度
    npu_group = str(npu_group)  # 将 npu_group 转为字符串格式

    # 根据 batch 是否是 ORCA 类型，设置不同的输出路径和命令
    if not batch.isORCA:  # 如果 batch 不是 ORCA 类型
        out_dir = f"../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}"  # 设置输出目录
        if not os.path.isdir(out_dir):  # 如果输出目录不存在
            os.mkdir(out_dir)  # 创建该目录
        # 构建命令字符串，执行 chakra.et_converter.et_converter 脚本
        cmd = f'python -m chakra.et_converter.et_converter --input_type LLM ' \
                f'--input_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}.txt ' \
                f'--output_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}/llm ' \
                f'--num_npus {npu_num} --num_dims 1 --num_passes 1'
    else:  # 如果 batch 是 ORCA 类型
        out_dir = f"../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}"  # 设置输出目录
        if not os.path.isdir(out_dir):  # 如果输出目录不存在
            os.mkdir(out_dir)  # 创建该目录
        # 构建命令字符串，执行 chakra.et_converter.et_converter 脚本
        cmd = f'python -m chakra.et_converter.et_converter --input_type LLM ' \
                f'--input_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}.txt ' \
                f'--output_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}/llm ' \
                f'--num_npus {npu_num} --num_dims 1 --num_passes 1'
    # 打印构建的命令（可以注释掉）
    # print(cmd)  
    cmd = cmd.split()  # 将命令字符串拆分为列表形式，以便传递给 subprocess.run
    start = time()  # 记录当前时间，开始执行命令
    subprocess.run(cmd, text=True)  # 执行外部命令
    end = time()  # 记录命令执行后的时间
    os.chdir(cwd)  # 执行完命令后，切换回原工作目录
    # 打印日志（可以注释掉）
    # print("generateGraph done")  
    return (end-start)*1000  # 返回执行时间，单位为毫秒
