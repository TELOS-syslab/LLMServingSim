import math
from functools import reduce
import re
from time import time
from .request import *
from .utils import *
import pandas as pd


# memory spec parameters
_dram_page_size = 512  # 内存页面大小，单位：字节
_dram_banks_per_ch = 32  # 每个内存通道的存储体数量
_gwrite_latency = 100  # 内存写入延迟
_gemv_latency = 184  # GEMV（矩阵乘法）延迟
# clock = 1GHz (same as LLM-Sim)  # 时钟频率，1 GHz（与LLM-Sim相同）

# model spec
E = 4096  # 模型的嵌入维度
n_tp = 4  # 模型的分片数
_nh = 32  # 头数
_dk = E / _nh  # 每个头的维度

# 估计多头自注意力（MHA）延迟的函数
def estimate_mha_latency(batch):
    E, _, _nh = get_config(batch.model)  # 获取模型的配置
    _dk = E / _nh  # 计算每个头的维度
    pim_times = []  # 存储延迟时间的列表
    for req in batch.requests:  # 遍历批次中的每个请求
        if not req.isInit:  # 如果请求不是初始化
            seq_len = req.input  # 获取输入序列长度

            _dk = E / _nh  # 重新计算每个头的维度
            _effective_e = E / n_tp  # 计算有效的E值
            # 计算MHA延迟
            kq_latency = 0  # key-query延迟
            lv_latency = 0  # logit-value延迟

            # key * query
            chunks = math.ceil(_effective_e / _dram_page_size)  # 计算内存块的数量
            tiles = math.ceil(seq_len / _dram_banks_per_ch)  # 计算tiles的数量
            kq_latency += chunks * _gwrite_latency  # 写入延迟
            kq_latency += chunks * tiles * _gemv_latency  # GEMV延迟

            # logit * value
            chunks = math.ceil(seq_len / _dram_page_size) * _nh  # 计算内存块的数量
            tiles = math.ceil(_dk / _dram_banks_per_ch)  # 计算tiles的数量
            lv_latency += chunks * _gwrite_latency  # 写入延迟
            lv_latency += chunks * tiles * _gemv_latency  # GEMV延迟

            pim_times.append(kq_latency)  # 将key-query延迟加入列表
            pim_times.append(lv_latency)  # 将logit-value延迟加入列表

    return pim_times  # 返回延迟时间列表

# 计算多个序列长度的总负载
def sum_load(seqlens):
    load = reduce(lambda acc, seq_len: acc + estimate_mha_latency(seq_len), seqlens, 0)  # 使用reduce累加计算负载
    return load  # 返回总负载

# 将新请求分配到多个通道，分配时考虑每个通道的负载
def distribute_requests(new_seq_lens, channels_seqlen, k):
    # 创建一个列表，用于存储每个通道的总负载
    channels_load = [sum_load(seqlens) for seqlens in channels_seqlen]
    
    for element in sorted(new_seq_lens, reverse=True):  # 按序列长度降序排列新请求
        min_sum_index = min(range(k), key=lambda i: channels_load[i])  # 找到负载最小的通道
        channels_seqlen[min_sum_index].append(element)  # 将请求分配给负载最小的通道
        channels_load[min_sum_index] += estimate_mha_latency(element)  # 更新该通道的负载
        
    return channels_seqlen  # 返回更新后的通道分配

# 添加PIM时间到文件中，按指定的格式保存
def addPIMtime(batch, npu_group, pim_times, pim_type):
    model = batch.model  # 获取模型名称
    batch_size = batch.batch_size  # 获取批次大小
    npu_group = str(npu_group)  # 将NPU组转为字符串
    input_len = batch.input  # 获取输入长度
    init_cnt = batch.output  # 获取输出长度
    parallel = 'hybrid'  # 并行策略设置为混合模式
    output_path = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}.txt"  # 设置输出路径

    with open(output_path, 'r') as f:  # 读取文件
        result = []
        for line in f.readlines():  # 按行读取
            split = re.findall(r'\S+', line)  # 用正则表达式提取每行的非空白字符
            result.append(split)  # 将提取的内容添加到列表中
        
    with open(output_path, 'w') as f:  # 重新写入文件
        if pim_type == 'local':  # 如果PIM类型是local
            f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")  # 写入ORCA类型和NPU组信息
        else:  # 如果PIM类型是其他
            f.write(f"PIM_POOL\t\tmodel_parallel_NPU_group: {npu_group}\n")  # 写入PIM_POOL类型和NPU组信息
        f.write(result[1][0]+'\n')  # 写入文件的第二行内容
        f.write(header())  # 写入头部信息

        # 写入正文
        tp_cnt = 0  # 初始化tensor transpose计数器
        mm_cnt = 0  # 初始化matmul计数器
        init = False  # 初始化标志

        for i in range(3, len(result)):  # 从第四行开始处理
            if "ATTENTION" not in result[i][0]:  # 如果当前行不包含"ATTENTION"
                if "tensor_transpose4d" in result[i][0]:  # 如果包含tensor_transpose4d
                    tp_cnt += 1  # tensor transpose计数器加1
                if tp_cnt == 4:  # 如果已经处理过4次tensor transpose
                    init = True  # 设置初始化标志
                if not init and 'matmul4d' in result[i][0]:  # 如果未初始化并且包含matmul4d
                    f.write(formatter(f'pim_{mm_cnt}_{i-3}', str(pim_times[mm_cnt]), *result[i][2:], parallel))  # 格式化写入
                    mm_cnt += 1  # matmul计数器加1
                else:
                    f.write(formatter(*result[i], parallel))  # 格式化写入其他内容
            else:
                f.write(formatter(' '.join(result[i]),'','','','','','','','','','', parallel))  # 格式化写入ATTENTION相关内容
                tp_cnt = 0  # 重置tensor transpose计数器
                init = False  # 重置初始化标志
                if "END" in result[i][1]:  # 如果包含"END"
                    mm_cnt = 0  # 重置matmul计数器

def subbatchInt(batch): 
    if len(batch.requests) == 1:  # 如果批次只有一个请求，直接返回该批次
        return [batch]
    if int(batch.output) == len(batch.requests):  # 如果输出等于请求数量，没有可能的重叠，直接返回该批次
        return [batch]
    # 将批次分为两部分
    reqs = batch.requests[:]  # 复制请求列表
    reqs = sorted(reqs, reverse=True, key=lambda x: x.input)  # 按请求的输入长度降序排序
    req1 = []  # 存储第一部分请求
    req2 = []  # 存储第二部分请求
    for i, req in enumerate(reqs):  # 遍历请求列表
        if i % 2 == 0:  # 偶数索引的请求放到req1中
            req1.append(req)
        else:  # 奇数索引的请求放到req2中
            req2.append(req)
    req1 = sorted(req1, key=lambda x: x.arrival)  # 按到达时间排序req1
    req2 = sorted(req2, key=lambda x: x.arrival)  # 按到达时间排序req2
    total_len = 0  # 初始化总长度
    init_cnt = 0  # 初始化初始化请求计数
    for req in req1:  # 遍历req1
        if req.isInit:  # 如果是初始化请求
            total_len += req.input  # 累加输入长度
            init_cnt += 1  # 初始化请求计数加1
        else:  # 如果不是初始化请求
            total_len += 1  # 计算非初始化请求长度
    # 为生成文本函数（generateText）准备的批次，仅需要有效的值（其他值设为0）
    batch1 = Batch(0, batch.model, str(total_len), str(init_cnt), '1', 0, 0, batch.evict, batch.load, True)  # 创建批次1
    batch1.requests.extend(req1)  # 将req1中的请求添加到批次1中
    total_len = 0  # 重置总长度
    init_cnt = 0  # 重置初始化请求计数
    for req in req2:  # 遍历req2
        if req.isInit:  # 如果是初始化请求
            total_len += req.input  # 累加输入长度
            init_cnt += 1  # 初始化请求计数加1
        else:  # 如果不是初始化请求
            total_len += 1  # 计算非初始化请求长度
    # KV缓存仅处理一次
    batch2 = Batch(0, batch.model, str(total_len), str(init_cnt), '1', 0, 0, 0, 0, True)  # 创建批次2
    batch2.requests.extend(req2)  # 将req2中的请求添加到批次2中
    return [batch1, batch2]  # 返回两个子批次

def mergeText(batch, subbatches, num_npus, npu_group):
    # 如果只有初始化阶段（不需要进行子批次处理）
    if len(subbatches) == 1:
        return

    npus_per_group = num_npus // npu_group  # 计算每个NPU组的NPU数量
    # 打开文本文件
    model = batch.model  # 获取模型名称
    batch_size = batch.batch_size  # 获取批次大小
    npu_group = str(npu_group)  # 将NPU组转换为字符串
    parallel = 'hybrid'  # 设置并行方式为hybrid（混合模式）

    input_len1 = int(subbatches[0].input)  # 获取子批次1的输入长度
    init_cnt1 = int(subbatches[0].output)  # 获取子批次1的初始化请求数量
    output_path1 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len1}_orca_n{npu_group}.txt"  # 设置输出路径1
    input_len2 = int(subbatches[1].input)  # 获取子批次2的输入长度
    init_cnt2 = int(subbatches[1].output)  # 获取子批次2的初始化请求数量
    output_path2 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len2}_orca_n{npu_group}.txt"  # 设置输出路径2
    input_len3 = int(batch.input)  # 获取原始批次的输入长度
    init_cnt3 = int(batch.output)  # 获取原始批次的初始化请求数量
    output_path3 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len3}_orca_n{npu_group}.txt"  # 设置输出路径3

    with open(output_path1, 'r') as f1:  # 读取输出路径1的文件
        b1 = []  # 存储文件内容
        for line in f1.readlines():  # 按行读取
            split = re.findall(r'\S+', line)  # 使用正则表达式提取每行的非空白字符
            b1.append(split)  # 将提取的内容添加到b1中

    with open(output_path2, 'r') as f2:  # 读取输出路径2的文件
        b2 = []  # 存储文件内容
        for line in f2.readlines():  # 按行读取
            split = re.findall(r'\S+', line)  # 使用正则表达式提取每行的非空白字符
            b2.append(split)  # 将提取的内容添加到b2中

    # result字典
    b = []  # 存储合并后的内容
    # 添加头部信息
    b.extend(b1[:3])  # 添加b1的前3行
    # 移除头部信息
    b1 = b1[3:]  # 移除b1的头部
    b2 = b2[3:]  # 移除b2的头部
    # 添加vllm
    i = 0
    while 'vllm' in b1[i][0]:  # 如果当前行包含"vllm"
        b.append(b1[i])  # 将该行添加到b中
        i += 1
    b1 = b1[i:]  # 移除已添加的vllm部分
    # 提取每层
    embd1, ln1, qkv1, attn1, proj1, res1, ffn11, gelu1, ffn21 = extractLayer(b1)  # 提取第一层
    embd2, ln2, qkv2, attn2, proj2, res2, ffn12, gelu2, ffn22 = extractLayer(b1)  # 提取第二层

    # 安排各层
    b.extend(embd1)  # 添加embd1到b
    b.extend(ln1)  # 添加ln1到b
    b.append(qkv1)  # 添加qkv1到b

    b.extend(embd2)  # 添加embd2到b
    b.extend(ln2)  # 添加ln2到b
    # 计算注意力（attn）
    attn_npu1 = {}  # 存储第一部分的注意力NPU
    attn_init1 = {}  # 存储第一部分初始化的注意力
    attn_npu2 = {}  # 存储第二部分的注意力NPU
    attn_init2 = {}  # 存储第二部分初始化的注意力

    # 对每一层的注意力进行处理
    for i, attn in enumerate(attn1):  # 遍历第一部分的注意力
        if i < init_cnt1:  # 如果是初始化请求
            if i % npus_per_group not in attn_init1:  # 如果该NPU组还没有存储该层
                attn_init1[i % npus_per_group] = [attn]  # 添加到初始化注意力字典
            else:
                attn_init1[i % npus_per_group].extend(attn)  # 否则，将当前注意力添加到该NPU组
        if i % npus_per_group not in attn_npu1:  # 如果该NPU组没有存储该层的注意力计算
            attn_npu1[i % npus_per_group] = sum([int(j[1]) for j in attn])  # 计算该组的注意力总量
        else:
            attn_npu1[i % npus_per_group] += sum([int(j[1]) for j in attn])  # 累加该组的注意力总量

    for i, attn in enumerate(attn2):  # 对第二部分的注意力进行相同的处理
        if i < init_cnt2:
            if i % npus_per_group not in attn_init2:
                attn_init2[i % npus_per_group] = [attn]
            else:
                attn_init2[i % npus_per_group].extend(attn)
        if i % npus_per_group not in attn_npu2:
            attn_npu2[i % npus_per_group] = sum([int(j[1]) for j in attn])
        else:
            attn_npu2[i % npus_per_group] += sum([int(j[1]) for j in attn])

    # 仅添加剩余的注意力计算时间，考虑与qkv2的重叠
    for i in range(npus_per_group):
        b.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init1:
            b.extend(attn_init1[i])
        if i in attn_npu1 and attn_npu1[i] > int(qkv2[1]) // npus_per_group:
            b.append(['attn_overlap_1', f'{attn_npu1[i] - int(qkv2[1]) // npus_per_group}', 'LOCAL', '0', 'REMOTE', '0', 'REMOTE', '0', 'NONE', '0', 'NONE'])
    b.append([f'ATTENTION END', '', '', '', '', '', '', '', '', '', ''])

    # 创建块
    block1 = [proj1, res1, *ln1, ffn11, *gelu1, ffn12, res1, *ln1, qkv1]
    block1_gemm = int(proj1[1]) + int(ffn11[1]) + int(ffn12[1]) + int(qkv1[1])
    block2 = [proj2, res2, *ln2, ffn21, *gelu2, ffn22, res2, *ln2, qkv2]
    block2_gemm = int(proj2[1]) + int(ffn21[1]) + int(ffn22[1]) + int(qkv2[1])

    # 添加剩余的注意力计算时间，考虑与gemms的重叠
    for i in range(npus_per_group):
        block1.append([f'ATTENTION {i}','','','','','','','','','',''])
        block2.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init1:
            block2.extend(attn_init1[i])
        if i in attn_npu1 and attn_npu1[i] > block2_gemm // npus_per_group:
            block2.append(['attn_overlap_1', f'{attn_npu1[i] - block2_gemm // npus_per_group}', 'LOCAL', '0', 'REMOTE', '0', 'REMOTE', '0', 'NONE', '0', 'NONE'])
        if i in attn_init2:
            block1.extend(attn_init2[i])
        if i in attn_npu2 and attn_npu2[i] > block1_gemm // npus_per_group:
            block1.append(['attn_overlap_1', f'{attn_npu2[i] - block1_gemm // npus_per_group}', 'LOCAL', '0', 'REMOTE', '0', 'REMOTE', '0', 'NONE', '0', 'NONE'])
    block1.append([f'ATTENTION END', '', '', '', '', '', '', '', '', '', ''])
    block2.append([f'ATTENTION END', '', '', '', '', '', '', '', '', '', ''])
    _, n_layer, _ = get_config(model)  # 获取层数

    # 重复N-1次
    for _ in range(n_layer - 1):
        b.extend(block1)
        b.extend(block2)

    # 结尾步骤
    end1 = [proj1, res1, *ln1, ffn11, *gelu1, ffn12, res1, *ln1]  # 结束步骤1
    end1_gemm = int(proj1[1]) + int(ffn11[1]) + int(ffn12[1])
    # 添加剩余的注意力计算时间，考虑与gemms的重叠
    for i in range(npus_per_group):
        end1.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init2:
            end1.extend(attn_init2[i])
        if i in attn_npu2 and attn_npu2[i] > end1_gemm // npus_per_group:
            end1.append(['attn_overlap_1', f'{attn_npu2[i] - end1_gemm // npus_per_group}', 'LOCAL', '0', 'REMOTE', '0', 'REMOTE', '0', 'NONE', '0', 'NONE'])
    end1.append([f'ATTENTION END', '', '', '', '', '', '', '', '', '', ''])

    end2 = [proj2, res2, *ln2, ffn21, *gelu2, ffn22, res2, *ln2]  # 结束步骤2
    b.extend(end1)
    b.extend(end2)

    layer_num = len(b) - 3  # 计算层数
    b[1][0] = str(layer_num)  # 设置层数

    # 存储合并后的结果
    with open(output_path3, 'w') as f:
        f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")
        f.write(b[1][0]+'\n')
        f.write(header())

        # 写入内容
        for i in range(3, len(b)):
            if "ATTENTION" not in b[i][0]:  # 如果不是注意力层
                name = '_'.join(b[i][0].split('_')[:-1])
                new_string = f'{name}_{i-3}'
                # 检查输入输出
                if i < len(b) - 1 and "ATTENTION" not in b[i+1][0]:
                    cur_output = int(b[i][7])
                    next_input = int(b[i+1][3])
                    if next_input > cur_output:
                        b[i][7] = str(next_input)
                    else:
                        b[i+1][3] = str(cur_output)
                f.write(formatter(new_string, *b[i][1:], parallel))
            else:
                f.write(formatter(*b[i], parallel))

def extractLayer(b1):  
    # 检查嵌入层（embd），取出前两项  
    embd = b1[:2]  
    # 检查层归一化（ln），直到遇到包含 'gemm' 的项  
    i = 2  
    ln = []  
    while 'gemm' not in b1[i][0]:  # 当当前行不包含 'gemm' 时，表示还未到达 'gemm' 层  
        ln.append(b1[i])  # 将当前行添加到 ln 列表  
        i+=1  # 移动到下一行  
    # 检查QKV层（qkv），取出当前行  
    qkv = b1[i]  
    i+=1  # 移动到下一行  
    # 检查注意力层（Attn），直到遇到包含 'END' 的项  
    attn = []  
    while 'END' not in b1[i][1]:  # 当当前行的第二项不包含 'END' 时，继续查找注意力层  
        temp = []  
        i+=1  # 移动到下一行  
        while 'ATTENTION' not in b1[i][0]:  # 如果当前行不包含 'ATTENTION'，继续查找  
            temp.append(b1[i])  # 将当前行添加到临时列表  
            i+=1  # 移动到下一行  
        attn.append(temp)  # 将临时注意力层添加到 attn 列表  
    i+=1  # 移动到下一行，跳过 'END' 行  
    # 检查投影层（proj），取出当前行  
    proj = b1[i]  
    i+=1  # 移动到下一行  
    # 检查残差连接层（res），取出当前行  
    res = b1[i]  
    i+=1  # 移动到下一行  
    # 检查第一部分前馈神经网络层（ffn1），直到遇到包含 'gemm' 的项  
    while 'gemm' not in b1[i][0]:  
        i+=1  # 移动到下一行  
    ffn1 = b1[i]  # 取出当前行作为 ffn1  
    i+=1  # 移动到下一行  
    # 检查 GELU 激活层（gelu），直到遇到包含 'gemm' 的项  
    gelu = []  
    while 'gemm' not in b1[i][0]:  
        gelu.append(b1[i])  # 将当前行添加到 gelu 列表  
        i+=1  # 移动到下一行  
    # 检查第二部分前馈神经网络层（ffn2），取出当前行  
    ffn2 = b1[i]  

    # 返回所有提取的层  
    return embd, ln, qkv, attn, proj, res, ffn1, gelu, ffn2  


def dataset_converter(input_file_path, output_file_path):  
    alpaca_data = pd.read_csv(input_file_path)  # 读取输入文件的 CSV 数据  

    # 转换数据，创建新的 DataFrame  
    alpaca_data_transformed = pd.DataFrame({  
        'input_toks': alpaca_data['seq_len'],  # 将原始数据中的 'seq_len' 列重命名为 'input_toks'  
        'output_toks': 1,  # 设置 'output_toks' 列为常数 1  
        'arrival_time_ns': 0  # 设置 'arrival_time_ns' 列为常数 0  
    })  

    alpaca_data_transformed.to_csv(output_file_path, sep='\t', index=False)  # 将转换后的数据保存为 TSV 格式  
    return  # 函数没有返回任何值  
