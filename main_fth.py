import os  # 导入os模块，用于操作系统相关功能
import subprocess  # 导入subprocess模块，用于执行子进程
import math  # 导入math模块，提供数学函数
from time import time  # 从time模块导入time函数，用于计时
import argparse  # 导入argparse模块，用于解析命令行参数
import pandas as pd  # 导入pandas模块，用于数据处理

from inference_serving.scheduler import *  # 导入调度器模块的所有内容
from inference_serving.request import *  # 导入请求模块的所有内容
from inference_serving.utils import *  # 导入工具模块的所有内容
from inference_serving.control import *  # 导入控制模块的所有内容
from inference_serving.kv_manage import *  # 导入kv管理模块的所有内容
from inference_serving.generate_graph import *  # 导入图生成模块的所有内容
from inference_serving.generate_text import *  # 导入文本生成模块的所有内容
from inference_serving.pim import *  # 导入PIM模块的所有内容


def main():
    cwd = os.getcwd()  # 获取当前工作目录
    astra_sim = os.path.join(cwd, "astra-sim")  # 拼接astra-sim路径
    os.chdir(astra_sim)  # 切换到astra-sim目录
    parser = argparse.ArgumentParser(description='LLM-Sim')  # 创建命令行参数解析器

    # 添加命令行参数
    parser.add_argument('--model_name', type=str, help='Name of the model', default='gpt2')  # 模型名称
    parser.add_argument('--npu_num', type=int, help='# of NPUs', default=16)  # NPU数量
    parser.add_argument('--max_batch', type=int, help='maximum size of the batch', default=0)  # 最大批次大小
    parser.add_argument('--batch_delay', type=int, help='batch delay', default=0)  # 批次延迟
    parser.add_argument('--scheduling', type=str, help='scheduling of the system', default='orca')  # 调度策略
    parser.add_argument('--parallel', type=str, help='parallelism', default='hybrid')  # 并行方式
    parser.add_argument('--npu_group', type=int, help='npu_group', default=1)  # NPU分组
    parser.add_argument('--npu_mem', type=int, help='npu memory', default=40)  # NPU内存大小（GB）
    parser.add_argument('--kv_manage', type=str, help='kv cache management', default='vllm')  # KV缓存管理策略
    parser.add_argument('--block_size', type=int, help='kv cache block size', default=8)  # KV缓存块大小
    parser.add_argument('--pim_type', type=str, help='PIM attached type', default=None)  # PIM类型
    parser.add_argument('--sub_batch', action='store_true', default=False, help='PIM sub-batch interleaving')  # 是否使用PIM子批次交错
    parser.add_argument('--dataset', type=str, help='dataset path', default=None)  # 数据集路径
    parser.add_argument('--network', type=str, help='network config path', default=None)  # 网络配置文件路径
    parser.add_argument('--output', type=str, help='output tsv path', default=None)  # 输出TSV文件路径
    parser.add_argument('--gen', action='store_false', default=True, help='skip initiation phase')  # 是否跳过初始化阶段
    parser.add_argument('--fast_run', action='store_true', default=False, help='skip compilation')  # 是否跳过编译阶段

    args = parser.parse_args()  # 解析命令行参数  可以将args改为parameters

    ################################################################################################

    model = args.model_name  # 获取模型名称
    npu_num = args.npu_num  # 获取NPU数量
    max_batch = args.max_batch if args.max_batch != 0 else float('inf')  # 设置最大批次大小，0表示无限大
    batch_delay = args.batch_delay  # 获取批次延迟
    scheduling = args.scheduling if args.scheduling == 'orca' else None  # 设置调度策略
    parallel = args.parallel  # 设置并行方式
    npu_group = args.npu_group  # 设置NPU分组
    npu_mem = args.npu_mem  # 设置NPU内存大小（GB）
    kv_manage = args.kv_manage  # 设置KV缓存管理策略
    block_size = args.block_size  # 设置KV缓存块大小
    pim_type = args.pim_type if args.pim_type in ['local', 'pool'] else None  # 设置PIM类型
    sub_batch = args.sub_batch  # 是否使用子批次
    dataset = args.dataset  # 获取数据集路径
    network_file = args.network  # 获取网络配置文件路径
    output_file = args.output  # 获取输出文件路径
    isInit = args.gen  # 是否初始化
    fast_run = args.fast_run  # 是否快速运行

    # 在模拟大规模语言模型（LLM）推理服务的系统中，我们使用 Astra-sim 作为模拟器，来模拟计算单元（NPU）的行为。当 PIM（Processing in Memory，存内计算）类型为 'pool' 时，PIM 被设计为额外的计算单元，与 NPU 一起协同工作。
    # 具体来说，'pool' 模式下的 PIM 不仅仅是存储器的扩展，而是被视为可以执行计算的单元。这意味着在模拟中，我们需要考虑 PIM 作为独立的计算资源。因此，为了在模拟中准确反映系统的计算能力，我们将 NPU 的数量加上 PIM 的数量。
    if pim_type == 'pool':
        npu_num += npu_num  # 如果PIM类型为pool，NPU数量翻倍

    ### 应该根据npu_group更改网络配置 ###
    ### 需要在网络json文件中有相同的维度 ###
    ### 目前有1到128个NPU的json文件，可以手动添加 ###
    if network_file == None:
        network_dim = int(math.log2(npu_group)) + 1  # 计算网络维度
        if npu_group == npu_num:  # 如果NPU分组等于NPU数量，表示使用流水线并行
            network_dim = 1
        if pim_type != 'pool':
            network = astra_sim + f"/inputs/network/analytical/fully_connected_{network_dim}d_{npu_num}.json"  # 设置网络配置文件路径
        else:
            network = astra_sim + f"/inputs/network/analytical/pim_pool_{npu_num}.json"  # 设置PIM池模式的网络配置文件路径
    else:
        network = astra_sim + "/inputs/network/analytical/" + network_file  # 使用指定的网络配置文件路径

    binary = astra_sim + "/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"  # 模拟器二进制文件路径
    system = astra_sim + "/inputs/system/sample_fully_connected_sys.txt"  # 系统配置文件路径
    memory = astra_sim + "/inputs/remote_memory/analytical/per_npu_memory_expansion.json"  # 内存配置文件路径

    ### 你需要在request/model_reference/中添加模型参考文件 ###
    ### 这将计算模型权重大小以管理内存约束 ###
    ################################################################################################

    start = time()  # 记录开始时间
    # 计算模拟器运行时延迟
    astra_time = 0  # Astra模拟器运行时间
    graph_time = 0  # 图生成时间
    compile_time = 0  # 编译时间
    simulate_time = 0  # 模拟时间
    orca_time = 0  # ORCA调度时间
    vllm_time = 0  # vLLM管理时间
    pim_time = 0  # PIM模拟时间
    subbatch_time = 0  # 子批次处理时间

    scheduler = Scheduler(model, max_batch, batch_delay, scheduling, parallel, npu_num, npu_group, npu_mem, kv_manage,
                          block_size, pim_type)  # 创建调度器实例

    if dataset != None:
        # 生成泊松分布的请求
        scheduler.generate(dataset, isInit=isInit)
    else:
        # 手动添加请求
        for i in range(16):  # 模型、序列长度、结束长度、到达时间
            scheduler.addRequest([model, 128, 129, 0])

    # 模拟器开始
    current = 0  # 系统当前时钟周期
    sys = 0  # 当前系统ID（NPU ID）
    id = 0  # 请求ID

    # 获取第一个请求
    first = scheduler.getRequest(current, [i for i in range(npu_num)])

    if npu_num >= 64 and 'neupims' in network:
        sub_batch = False  # 大规模系统不使用子批次
    if sub_batch:
        sb_st = time()  # 子批次处理开始时间
        bats = subbatchInt(first)  # 对请求进行子批次处理
        sb_ed = time()  # 子批次处理结束时间
        subbatch_time += sb_ed - sb_st  # 累加子批次处理时间
    else:
        bats = [first]  # 不使用子批次

    for bat in bats:
        # 使用codelet编译并生成完整的请求文本
        comp, sim = generateText(bat, parallel, npu_group, fast_run, network_file)
        compile_time += comp  # 累加编译时间
        simulate_time += sim  # 累加模拟时间

        # PIM激活
        if pim_type != None:
            # 调用PIM模拟器并获取结果
            # 生成阶段的GEMV的pim_times
            pim_st = time()  # PIM模拟开始时间
            addPIMtime(bat, npu_group, estimate_mha_latency(bat), pim_type)  # 添加PIM时间
            pim_ed = time()  # PIM模拟结束时间
            pim_time += pim_ed - pim_st  # 累加PIM模拟时间

    # 调度子批次
    if sub_batch:
        sb_st = time()  # 子批次合并开始时间
        mergeText(first, bats, npu_num, npu_group)  # 合并子批次文本
        sb_ed = time()  # 子批次合并结束时间
        subbatch_time += sb_ed - sb_st  # 累加子批次合并时间

    # 生成Chakra图
    # 编译器需要将神经网络模型算子映射到底层存算单元
    graph = generateGraph(first, parallel, npu_group, npu_num)
    graph_time += graph  # 累加图生成时间

    # 设置第一个工作负载文件
    workload = getWorkload(first, parallel, npu_group)
    # 运行子进程
    args = [binary, "--workload-configuration=" + workload, "--system-configuration=" + system,
            "--network-configuration=" + network, "--remote-memory-configuration=" + memory]
    astra_st = time()  # Astra模拟器启动时间
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)  # 启动Astra模拟器子进程

    # 计算模拟器的吞吐量
    throughput = []
    prompt_th = 0  # 每秒平均提示吞吐量
    gen_th = 0  # 每秒平均生成吞吐量
    last_log = 0  # 上次记录的时间
    FREQ = 1000000000  # 1 GHz
    INTERVAL = 500000000  # 0.5秒
    RATIO = FREQ // INTERVAL  # 计算比率
    total_prompt = 0  # 总提示吞吐量
    total_gen = 0  # 总生成吞吐量
    total_latency = 0  # 总延迟

    while True:
        out = readWait(p)  # 等待并读取子进程输出
        astra_ed = time()  # Astra模拟器结束时间
        astra_time += (astra_ed - astra_st) * 1000  # 累加Astra模拟器运行时间（毫秒）
        out_dict = parseOutput(out[-2])  # 解析输出

        if out_dict != None:
            sys = out_dict['sys']  # 更新系统ID
            id = out_dict['id']  # 更新请求ID
            current = out_dict['cycle']  # 更新当前时钟周期

        # 获取可用请求
        # 新的请求总是从sys[0]开始
        if sys == 0:
            new_req = scheduler.getRequest(current, [sys])  # 获取新的请求

            if new_req != None:
                if sub_batch:
                    sb_st = time()  # 子批次处理开始时间
                    bats = subbatchInt(new_req)  # 对请求进行子批次处理
                    sb_ed = time()  # 子批次处理结束时间
                    subbatch_time += sb_ed - sb_st  # 累加子批次处理时间
                else:
                    bats = [new_req]

                for bat in bats:
                    # 编译并生成完整的请求文本
                    comp, sim = generateText(bat, parallel, npu_group, fast_run, network_file)
                    compile_time += comp  # 累加编译时间
                    simulate_time += sim  # 累加模拟时间

                    # PIM激活
                    if pim_type != None:
                        # 调用PIM模拟器并获取结果
                        # 生成阶段的GEMV的pim_times
                        pim_st = time()  # PIM模拟开始时间
                        addPIMtime(bat, npu_group, estimate_mha_latency(bat), pim_type)  # 添加PIM时间
                        pim_ed = time()  # PIM模拟结束时间
                        pim_time += pim_ed - pim_st  # 累加PIM模拟时间

                # 调度子批次
                if sub_batch:
                    sb_st = time()  # 子批次合并开始时间
                    mergeText(new_req, bats, npu_num, npu_group)  # 合并子批次文本
                    sb_ed = time()  # 子批次合并结束时间
                    subbatch_time += sb_ed - sb_st  # 累加子批次合并时间
                # 生成Chakra图
                graph = generateGraph(new_req, parallel, npu_group, npu_num)
                graph_time += graph  # 累加图生成时间
        # 对于其他系统
        else:
            new_req = scheduler.getInflight(id + 1, sys)  # 获取正在进行的请求

        # 如果没有请求要发出，则等待
        if new_req == None:
            # 检查最新的正在进行的请求
            astra_st = time()  # Astra模拟器等待开始时间
            writeFlush(p, "pass")  # 发送"pass"指令
        else:
            # 发送工作负载到Astra模拟器
            if new_req != None:
                # 设置工作负载文件
                workload = getWorkload(new_req, parallel, npu_group)
                # print(f"sys[{sys}]: {workload}")
                astra_st = time()  # Astra模拟器运行开始时间
                writeFlush(p, workload)  # 发送工作负载文件路径

        # 检查是否需要记录吞吐量
        if current > last_log + INTERVAL:
            # 存储提示阶段的吞吐量
            throughput.append((prompt_th * RATIO, gen_th * RATIO))
            last_log += INTERVAL  # 更新上次记录的时间
            print(f"[{last_log / FREQ}s] Avg Throughput: propmt: {prompt_th * RATIO}, generation: {gen_th * RATIO}")
            prompt_th = 0  # 重置提示吞吐量
            gen_th = 0  # 重置生成吞吐量
        # 检查请求是否完成
        prompt_t, gen_t = scheduler.addDone(id, sys, current)
        # 将tokens添加到吞吐量中
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        gen_th += gen_t  # 注意这里gen_th被加了两次，可能需要检查代码
        total_gen += gen_t

        if scheduler.isRequestEmpty():
            throughput.append((prompt_th * RATIO, gen_th * RATIO))
            last_log += INTERVAL
            print(f"[{last_log / FREQ}s] Avg Throughput: propmt: {prompt_th * RATIO}, generation: {gen_th * RATIO}")
            print("---------------------------")
            print("Exiting The Simulator")
            if scheduler.weight == scheduler.used_mem:
                print("Memory Is All Freed")
            else:
                print("Unfreed Memory Exists")
            astra_st = time()  # Astra模拟器退出开始时间
            writeFlush(p, "exit")  # 发送"exit"指令
            break

    # 检查所有请求是否已完成
    checkEnd(p)
    astra_ed = time()  # Astra模拟器退出结束时间
    astra_time += (astra_ed - astra_st) * 1000  # 累加Astra模拟器运行时间
    end = astra_ed  # 记录结束时间

    # 打印吞吐量结果
    scheduler.printResult()
    total_latency = current / FREQ  # 计算总延迟
    print('---------------------------')
    print('Throughput Results')
    print('---------------------------')
    print(f"Total prompts: {total_prompt} tokens/s")
    print(f"Total Generation: {total_gen} tokens/s")
    print(f"Throughput per {1 / RATIO} sec: {throughput}")
    print(f"Total clocks: {current} ticks")
    print(f"Total latency: {total_latency} s")
    print(f"Average throughput: prompt: {total_prompt / total_latency} generation: {total_gen / total_latency}")
    print('---------------------------')

    # 打印模拟时间
    orca_time = round(scheduler.orca * 1000, 3)  # ORCA时间（毫秒）
    vllm_time = round(scheduler.vllm * 1000, 3)  # vLLM时间（毫秒）
    astra_time = round(astra_time, 3)  # Astra模拟器时间（毫秒）
    graph_time = round(graph_time, 3)  # 图生成时间（毫秒）
    total_time = round((end - start) * 1000, 3)  # 总模拟时间（毫秒）
    compile_time = round(compile_time / 1000000, 3)  # 编译时间（毫秒）
    simulate_time = round(simulate_time / 1000000, 3)  # 模拟时间（毫秒）
    pim_time = round(pim_time * 1000, 3)  # PIM时间（毫秒）
    subbatch_time = round(subbatch_time * 1000, 3)  # 子批次时间（毫秒）
    execution_engine = compile_time + simulate_time + pim_time  # 执行引擎总时间
    scheduler_time = round(total_time - compile_time - simulate_time - graph_time - astra_time - pim_time, 3)  # 调度器时间
    print('Simulation Time (ms)')
    print('---------------------------')
    print(f"Total execution engine time: {execution_engine}")
    print(f"Total graph time: {graph_time}")
    print(f"Total astra time: {astra_time}")
    print(f"Total scheduler time: {scheduler_time}")
    print(f"Total simulation time: {total_time}")

    # 存储输出TSV文件
    if output_file != None:
        os.chdir(cwd)  # 切换到初始工作目录
        # 存储吞吐量数据
        time_durations = [i * (1 / RATIO) for i in range(1, len(throughput) + 1)]  # 时间间隔列表
        prompt_throughputs = [item[0] for item in throughput]  # 提示吞吐量列表
        generation_throughputs = [item[1] for item in throughput]  # 生成吞吐量列表

        df = pd.DataFrame({
            'time_duration': time_durations,
            'prompt_throughput': prompt_throughputs,
            'generation_throughput': generation_throughputs
        })

        output_path = output_file + '-throughput.tsv'  # 设置吞吐量输出文件路径
        df.to_csv(output_path, sep='\t', index=False)  # 保存吞吐量数据为TSV文件

        # 存储模拟时间数据
        simulation_times = {
            'execution_engine': execution_engine,
            'graph_converter': graph_time,
            'astra-sim': astra_time,
            'scheduler': scheduler_time,
            'total_simulation_time': total_time
        }

        df = pd.DataFrame([simulation_times])

        output_path = output_file + '-simulation-time.tsv'  # 设置模拟时间输出文件路径
        df.to_csv(output_path, sep='\t', index=False)  # 保存模拟时间数据为TSV文件


if __name__ == "__main__":
    main()  # 程序入口
