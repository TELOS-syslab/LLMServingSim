import pandas as pd  # 导入 pandas 库，用于数据处理
from time import time  # 导入 time 库，用于获取时间戳

from .request import *  # 导入 request 模块，包含请求相关的类和方法
from .utils import *  # 导入 utils 模块，包含工具函数
from .control import *  # 导入 control 模块，包含控制逻辑
from .kv_manage import *  # 导入 kv_manage 模块，包含键值存储管理
from .generate_graph import *  # 导入 generate_graph 模块，包含图生成相关功能
from .generate_text import *  # 导入 generate_text 模块，包含文本生成相关功能
from .pim import *  # 导入 pim 模块，包含 PIM 相关操作

# 定义一个调度类，用于调度 Astra-sim 请求
class Scheduler:
    def __init__(self, model, max_batch, batch_delay, scheduling, parallel, npu_num, npu_group, npu_mem, kv_manage, block_size, pim_type):
        # 初始化调度器
        self.model = model  # 模型名称
        self.max_batch = max_batch  # 每次最大批次大小
        self.batch_delay = batch_delay  # 批次间的延迟时间
        self.scheduling = scheduling  # 调度方法（例如 ORCA）
        self.parallel = parallel  # 是否并行
        self.npu_num = npu_num  # NPU 数量
        self.npu_group = npu_group  # NPU 分组
        # 以下三个列表按照请求到达时间排序
        self.request = []  # 请求队列
        self.inflight = []  # 执行中的批次
        self.done = []  # 已完成的请求
        self.reqIDs = -1  # 请求 ID 计数器
        self.batchIDs = -1  # 批次 ID 计数器
        # 假设所有 NPU 使用相同大小的内存
        self.npu_mem = npu_mem * 1000000000  # 将内存大小转为字节

        # 内存模型与 PIM 相关
        self.pim_type = pim_type  # PIM 类型（如 'pool'）
        if pim_type != 'pool':  # 如果不是池化模式
            self.weight = get_weight(model, npu_num)  # 假设权重已经加载
            self.kv_npu = self.npu_num  # 使用的 NPU 数量
        else:
            self.weight = 0  # 如果是池化模式，权重为 0
            self.kv_npu = self.npu_num // 2  # 假设权重存储在本地内存中，KV 缓存存在 PIM 中
        self.used_mem = self.weight  # 当前已使用的内存
        self.kv_manage = kv_manage  # KV 管理策略

        # vLLM 相关
        self.block_size = block_size  # vLLM 块大小

        self.orca = 0  # ORCA 调度时间统计
        self.vllm = 0  # vLLM 生成时间统计

    # 根据泊松分布生成请求
    def generate(self, path, isInit=True):
        data = pd.read_csv(path, sep='\t')  # 从文件中读取数据
        for index, row in data.iterrows():  # 遍历数据的每一行
            input_length = row['input_toks']  # 输入的 token 数
            output_length = row['input_toks'] + row['output_toks']  # 输出的 token 数
            arrival_time_ns = row['arrival_time_ns']  # 请求到达时间（纳秒）
            if index == 0:  # 对于第一个请求，强制将到达时间设为 0
                arrival_time_ns = 0
            # 添加请求到请求队列
            self.addRequest([self.model, input_length, output_length, arrival_time_ns], isInit=isInit)
        return

    # 批处理时使用，用于 vLLM 仅在初始化阶段
    def getBatchKV(self, batch_req, batch_len):
        batch_kv_size = 0  # 初始化 KV 大小为 0
        if self.kv_manage == 'max':  # 如果 KV 管理策略为 'max'
            if self.model == 'gpt2':  # 对于 GPT-2 模型
                batch_kv_size += batch_len * get_kv(self.model, 1024, self.kv_npu)  # 计算 KV 缓存大小
            elif self.model == 'gpt3-175b':  # 对于 GPT-3 175b 模型
                batch_kv_size += batch_len * get_kv(self.model, 4096, self.kv_npu)  # 计算 KV 缓存大小
            else:
                print("Error: Need to add max length of the model")  # 错误处理
        elif self.kv_manage == 'pow2':  # 如果 KV 管理策略为 'pow2'
            for i in range(batch_len):  # 遍历每个请求
                batch_kv_size += get_kv(self.model, batch_req[i].output * 2, self.kv_npu)  # 计算 KV 缓存大小
        elif self.kv_manage == 'oracle':  # 如果 KV 管理策略为 'oracle'
            for i in range(batch_len):  # 遍历每个请求
                batch_kv_size += get_kv(self.model, batch_req[i].output, self.kv_npu)  # 计算 KV 缓存大小
        elif self.kv_manage == 'vllm':  # 如果 KV 管理策略为 'vllm'
            for i in range(batch_len):  # 遍历每个请求
                num_blocks = batch_req[i].input // self.block_size + 1  # 计算所需的块数，包括当前迭代生成的 KV 缓存
                batch_kv_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)  # 计算 KV 缓存大小
        return batch_kv_size  # 返回计算的 KV 缓存大小

    # 获取应该添加的 KV 块的大小，用于 vLLM 生成阶段
    # 还会检查被驱逐的请求并包含其 KV 缓存
    def getBlockKV(self, batch_req, batch_len):
        block_kv_size = 0  # 初始化 KV 块大小为 0
        for i in range(batch_len):  # 遍历每个请求
            if batch_req[i].evict or batch_req[i].isInit:  # 如果是需要驱逐的请求或初始化请求
                num_blocks = batch_req[i].input // self.block_size + 1  # 计算需要的块数，包括当前迭代生成的 KV 缓存
                block_kv_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)  # 计算 KV 块大小
            else:
                num_before = (batch_req[i].input - 1) // self.block_size + 1  # 计算之前需要的块数
                num_after = batch_req[i].input // self.block_size + 1  # 计算当前需要的块数，包括当前迭代生成的 KV 缓存
                if num_after > num_before:  # 如果当前需要的块数比之前多
                    block_kv_size += get_kv(self.model, self.block_size, self.kv_npu)  # 增加一个块的 KV 缓存大小
        return block_kv_size  # 返回 KV 块大小
        
    # 获取应该被驱逐的 KV 缓存的大小
    def getEvictKV(self, req):
        evict_size = 0  # 初始化驱逐的 KV 缓存大小为 0
        # 当前输入加 1 还没有加载
        num_blocks = (req.input - 1) // self.block_size + 1  # 计算当前输入需要的块数
        evict_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)  # 计算 KV 缓存的大小
        return evict_size  # 返回 KV 缓存的大小

    # 内存加载方法
    def memLoad(self, size):
        # 如果加载的内存大小超过 NPU 可用内存
        # if self.used_mem + size > self.npu_mem:
            # print("ERROR: memLoad: no memory to load")
        # print(f"used: {self.used_mem} load: {size}", end=' ')
        self.used_mem += size  # 更新已使用的内存
        # print(f"after: {self.used_mem}")

    # 内存卸载方法
    def memStore(self, size):
        # 如果卸载的内存大小小于模型权重
        # if self.used_mem - size < self.weight:
            # print("ERROR: memStore: no memory to unload")
        # print(f"used: {self.used_mem} remove: {size}", end=' ')
        self.used_mem -= size  # 更新已使用的内存
        # print(f"after: {self.used_mem}")

    # 批量请求调度方法
    # TODO: 考虑输出长度
    def batch(self, current, sys):
        if self.scheduling == None:  # 如果没有指定调度方法
            delay_time = self.request[0].arrival + self.batch_delay  # 计算批次的延迟时间
            batch_req = [req for req in self.request if req.arrival <= current]  # 获取当前时间之前到达的请求
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch  # 批次长度（不超过最大批次）

            # 如果没有请求或需要更多时间等待
            if batch_len == 0 or (current < delay_time and batch_len != 0 and batch_len < self.max_batch):
                return None  # 无法生成批次，返回 None
            
            # 生成批次并继续处理
            batch_req = batch_req[:batch_len]

            # 检查内存
            for i in range(batch_len, -1, -1):  # 逆序遍历批次
                kv_size = self.getBatchKV(batch_req, i)  # 获取 KV 缓存大小
                if kv_size <= (self.npu_mem - self.used_mem):  # 如果内存足够
                    batch_len = i  # 更新批次长度
                    break
            # 如果没有足够的请求可以批量处理
            if batch_len == 0:
                return None
            batch_req = batch_req[:batch_len]  # 截取最终的请求列表
            self.memLoad(kv_size)  # 加载 KV 缓存

            # 从请求队列中删除已批处理的请求
            for _ in range(batch_len):
                del self.request[0]

            # 获取最大输入长度
            max_len = 0
            for req in batch_req:
                if max_len < req.input:
                    max_len = req.input  # 更新最大输入长度

            # 创建批次
            # TODO: 假设所有请求的输出长度相同
            batch = Batch(self.getBatchID(), batch_req[0].model, str(max_len), str(batch_req[0].output), str(batch_len), current, kv_size)
            # 将已经触发的系统添加到批次中
            batch.fired.extend(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)  # 将批次添加到执行中的批次列表
            # print(sys)
            
            return batch

        elif self.scheduling == 'orca':  # 如果调度方法是 ORCA
            # 检查并行执行的批次数量是否已达限制
            if len(self.inflight) >= self.npu_group:  # 如果执行中的批次数量超过 NPU 分组数量
                # 等待批次完成
                return None
            orca_start = time()  # 记录 ORCA 调度开始时间
            batch_req = [req for req in self.request if req.arrival <= current]  # 获取当前时间之前到达的请求
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch  # 批次长度（不超过最大批次）

            if batch_len == 0:
                return None  # 没有请求可用，返回 None

            # 生成批次并继续处理
            batch_req = batch_req[:batch_len]

            init_req = [req for req in batch_req if req.isInit]  # 获取初始化请求
            init_len = len(init_req)  # 初始化请求的数量

            kv_size = 0  # 初始化 KV 大小
            evict_size = 0  # 初始化驱逐的 KV 大小
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            if self.kv_manage != 'vllm':  # 如果 KV 管理策略不是 vLLM
                orca_start = time()  # 记录 ORCA 调度开始时间
                possible_init = 0  # 初始化可能的请求数量
                for i in range(init_len, -1, -1):  # 逆序遍历初始化请求
                    kv_size = self.getBatchKV(init_req, i)  # 获取 KV 缓存大小
                    if kv_size <= (self.npu_mem - self.used_mem):  # 如果内存足够
                        possible_init = i  # 更新可能的初始化请求数量
                        break

                # 如果没有足够的内存来批量处理初始化阶段的请求
                if possible_init == 0:
                    batch_len -= init_len  # 减去初始化请求的数量
                else:
                    batch_len -= init_len - possible_init  # 更新批次长度
                orca_end = time()  # 记录 ORCA 调度结束时间
                self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间

            # 生成阶段和 vLLM 处理
            elif self.kv_manage == 'vllm':  # 如果 KV 管理策略是 vLLM
                vllm_start = time()  # 记录 vLLM 处理开始时间
                gen_req = [req for req in batch_req if not req.isInit]  # 获取非初始化请求
                # 检查是否有请求需要增加块大小
                temp_len = batch_len
                for i in range(batch_len, -1, -1):  # 逆序遍历批次
                    kv_size = self.getBlockKV(batch_req, i)  # 获取 KV 块大小
                    if kv_size <= (self.npu_mem - self.used_mem):  # 如果内存足够
                        temp_len = i  # 更新批次长度
                        break

                # 如果没有足够的内存来批量处理
                while temp_len == 0:
                    # 逐个抢占请求直到有足够的内存
                    if len(gen_req) == 0:
                        return None
                    
                    # 检查是否有已经驱逐的请求
                    if gen_req[-1].evict:
                        gen_req = gen_req[:-1]  # 删除驱逐的请求
                        continue

                    # 否则，驱逐最后一个请求并更新内存
                    evict_size = self.getEvictKV(gen_req[-1])  # 获取驱逐的 KV 缓存大小
                    gen_req[-1].evict = True  # 标记请求为驱逐
                    gen_req = gen_req[:-1]  # 删除该请求
                    # print("eviction")
                    self.memStore(evict_size)  # 卸载内存

                    if len(gen_req) < batch_len:
                        batch_len = len(gen_req)  # 更新批次长度

                    # 检查是否可以批量处理
                    for i in range(batch_len, -1, -1):  # 逆序遍历批次
                        kv_size = self.getBlockKV(batch_req, i)  # 获取 KV 块大小
                        if kv_size <= (self.npu_mem - self.used_mem):  # 如果内存足够
                            temp_len = i  # 更新批次长度
                            break
                batch_len = temp_len  # 更新最终的批次长度
                vllm_end = time()  # 记录 vLLM 处理结束时间
                self.vllm += vllm_end - vllm_start  # 更新 vLLM 处理的总时间

            orca_start = time()  # 记录 ORCA 调度开始时间
            batch_req = batch_req[:batch_len]  # 截取最终的请求列表
            load_size = 0  # 初始化加载的 KV 缓存大小
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            # 从请求队列中删除已批处理的请求
            for req in batch_req:
                orca_start = time()  # 记录 ORCA 调度开始时间
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]  # 删除请求
                        break
                orca_end = time()  # 记录 ORCA 调度结束时间
                self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间

                if req.evict:  # 如果请求被标记为驱逐
                    vllm_start = time()  # 记录 vLLM 处理开始时间
                    load_size += self.getEvictKV(req)  # 加载驱逐的 KV 缓存
                    req.evict = False  # 标记请求不再驱逐
                    vllm_end = time()  # 记录 vLLM 处理结束时间
                    self.vllm = vllm_end - vllm_start  # 更新 vLLM 处理的总时间

            orca_start = time()  # 记录 ORCA 调度开始时间
            # 加载内存
            if kv_size > 0:
                self.memLoad(kv_size)  # 加载 KV 缓存
            
            total_len = 0  # 初始化总输入长度
            init_cnt = 0  # 初始化初始化请求的计数
            for req in batch_req:  # 遍历所有批次请求
                if req.isInit:  # 如果是初始化请求
                    total_len += req.input  # 更新总输入长度
                    init_cnt += 1  # 更新初始化请求的计数
                else:
                    total_len += 1  # 更新总输入长度

            # 创建批次
            # 输出不重要，始终为 1 次迭代
            # 批次数也为 1
            batch = Batch(self.getBatchID(), batch_req[0].model, str(total_len), str(init_cnt), '1', current, kv_size, evict_size, load_size, True)
            # 将已经触发的系统添加到批次中
            batch.fired.extend(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)  # 将批次添加到执行中的批次列表
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return batch  # 返回生成的批次

        
    # 如果可能，批量处理请求。返回批次并添加到 inflight 列表
    def getRequest(self, current, sys): # sys 应该是列表
        orca_start = time()  # 记录 ORCA 调度开始时间
        if len(self.request) != 0 and self.request[0].arrival <= current:  # 如果有请求且第一个请求的到达时间小于等于当前时间
            batch = self.batch(current, sys)  # 调用 batch 函数处理请求并生成批次
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return batch  # 返回生成的批次
        else:
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return None  # 没有可处理的请求，返回 None

    # 获取 inflight 中的请求
    def getInflight(self, batch_id, sys): # sys 是整数
        orca_start = time()  # 记录 ORCA 调度开始时间
        if len(self.inflight) == 0:  # 如果 inflight 列表为空
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return None  # 没有 inflight 请求，返回 None
        else:
            batch = None  # 初始化批次为 None
            # 查找指定的批次
            for b in self.inflight:
                if b.batch_id == batch_id:  # 如果找到匹配的批次 ID
                    batch = b  # 更新 batch
            if batch == None:  # 如果没有找到匹配的批次
                orca_end = time()  # 记录 ORCA 调度结束时间
                self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
                return None  # 返回 None
            # 检查该批次是否已在系统中执行过
            if sys in batch.fired:  # 如果该系统已经执行过
                orca_end = time()  # 记录 ORCA 调度结束时间
                self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
                return None  # 返回 None
            else:
                batch.fired.append(sys)  # 将当前系统添加到已执行系统列表
                orca_end = time()  # 记录 ORCA 调度结束时间
                self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
                return batch  # 返回匹配的批次

    # 获取新的请求 ID
    def getReqID(self):
        self.reqIDs += 1  # 请求 ID 自增
        return self.reqIDs  # 返回新的请求 ID

    # 获取新的批次 ID
    def getBatchID(self):
        self.batchIDs += 1  # 批次 ID 自增
        return self.batchIDs  # 返回新的批次 ID

    # 添加一个请求
    def addRequest(self, req, isInit=True):
        orca_start = time()  # 记录 ORCA 调度开始时间
        new = [self.getReqID()]  # 获取新的请求 ID
        new_req = Request(*(new + req), isInit=isInit)  # 创建新的请求对象
        self.request.append(new_req)  # 将请求添加到请求队列
        orca_end = time()  # 记录 ORCA 调度结束时间
        self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
        return  # 无返回

    # 从 inflight 中移除批次，添加到 done 列表
    def addDone(self, id, sys, finish):
        prompt_t = 0  # 初始化 prompt 时间
        gen_t = 0  # 初始化生成时间
        orca_start = time()  # 记录 ORCA 调度开始时间
        if len(self.inflight) == 0:  # 如果 inflight 列表为空
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return 0, 0  # 返回 0, 0
        batch = None  # 初始化批次为 None
        # 查找指定的批次
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:  # 如果找到匹配的批次 ID
                batch = b  # 更新 batch
                idx = i  # 记录批次索引
        # 如果没有找到批次，返回 0, 0
        if batch == None:
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return 0, 0  # 返回 0, 0
        # 如果系统已经完成
        if sys in batch.end:
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return 0, 0  # 返回 0, 0
        else:
            batch.end.append(sys)  # 将当前系统添加到已完成系统列表
            # 检查所有 NPU 是否都已完成
            for i in range(self.npu_num):
                if i not in batch.end:  # 如果有 NPU 没有完成
                    orca_end = time()  # 记录 ORCA 调度结束时间
                    self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
                    return 0, 0  # 返回 0, 0

        if self.scheduling == None:  # 如果没有调度方法
            batch.addLatency(finish)  # 添加延迟
            self.memStore(batch.kv_size)  # 卸载 KV 缓存
            # 将批次中的所有请求添加到 done 列表并删除批次
            self.done.extend(batch.requests)

        elif self.scheduling == 'orca':  # 如果调度方法是 orca
            pool = []  # 初始化请求池
            for req in batch.requests:  # 遍历批次中的所有请求
                if req.isInit:  # 如果是初始化请求
                    req.isInit = False  # 将请求标记为非初始化
                    prompt_t += req.input  # 更新 prompt 时间
                else:
                    gen_t += 1  # 更新生成时间

                req.input += 1  # 增加输入长度
                # 检查请求是否已完成
                if req.output <= req.input:  # 如果输出长度小于等于输入长度
                    kv_size = self.getEvictKV(req)  # 获取驱逐的 KV 缓存大小
                    self.memStore(kv_size)  # 卸载 KV 缓存
                    req.addLatency(finish)  # 添加延迟
                    self.done.append(req)  # 将请求添加到 done 列表
                else:
                    pool.append(req)  # 将未完成的请求添加到池中
            # 将请求池返回到请求队列前端
            self.request = pool + self.request

        del self.inflight[idx]  # 删除已完成的批次
        del batch  # 删除批次对象
        orca_end = time()  # 记录 ORCA 调度结束时间
        self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
        return prompt_t, gen_t  # 返回 prompt 和生成时间

    # 打印 done 列表中的结果
    def printResult(self):
        for i in self.done:  # 遍历所有已完成的请求
            print(i)  # 打印请求
        return  # 无返回

    # 检查所有请求是否已完成
    def isRequestEmpty(self):
        orca_start = time()  # 记录 ORCA 调度开始时间
        if len(self.request) == 0 and len(self.inflight) == 0:  # 如果请求队列和 inflight 列表都为空
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return True  # 返回 True
        else:
            orca_end = time()  # 记录 ORCA 调度结束时间
            self.orca += orca_end - orca_start  # 更新 ORCA 调度的总时间
            return False  # 返回 False
