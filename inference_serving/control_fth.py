def readWait(p):  # 定义函数 readWait，接受一个进程对象 p 作为参数
    # print("waiting astra-sim")  # 输出 "waiting astra-sim"（此行被注释掉，暂时不执行）
    out = [""]  # 初始化列表 out，包含一个空字符串，用于存储从进程输出读取的行
    while out[-1] != "Waiting\n" and out[-1] != "Checking Non-Exited Systems ...\n":  # 循环直到输出最后一行是 "Waiting\n" 或 "Checking Non-Exited Systems ..."
        out.append(p.stdout.readline())  # 从进程的标准输出中读取一行，并将其添加到 out 列表中
        p.stdout.flush()  # 刷新标准输出缓冲区，确保数据被输出到终端或文件
    # for i in out:  # 注释掉的代码，若启用则遍历输出的每一行
    #     if i != "Waiting\n":  # 如果当前行不是 "Waiting\n"
    #         print(i, end='')  # 打印该行
    return out  # 返回最终的输出列表 out，包含读取的所有行

def checkEnd(p):  # 定义函数 checkEnd，接受一个进程对象 p 作为参数
    out = ["", ""]  # 初始化列表 out，包含两个空字符串，存储进程输出的行
    while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":  # 循环直到倒数第二行是 "All Request Has Been Exited\n" 或 "ERROR: Some Requests Remain\n"
        out.append(p.stdout.readline())  # 从进程的标准输出中读取一行并追加到 out 列表中
        p.stdout.flush()  # 刷新标准输出缓冲区，确保数据被输出
    for i in out[4:]:  # 从列表 out 的第 4 项开始，遍历剩下的所有行
        print(i, end='')  # 打印每一行，去掉结尾的换行符
    return out  # 返回最终的输出列表 out，包含读取的所有行

def writeFlush(p, input):  # 定义函数 writeFlush，接受进程对象 p 和输入字符串 input 作为参数
    p.stdin.write(input + '\n')  # 将输入字符串写入进程的标准输入流，并添加换行符
    p.stdin.flush()  # 刷新标准输入缓冲区，确保数据被写入进程
    return  # 函数结束，不返回任何值

def parseOutput(output):  # 定义函数 parseOutput，接受一个字符串 output 作为参数
    if 'cycles' in output:  # 判断字符串中是否包含 'cycles'（如果包含则继续解析）
        sp = output.split()  # 将 output 字符串按空格分割成列表 sp
        sys = int(sp[0].split('[')[1].split(']')[0])  # 从 sp 的第一个元素中提取系统 ID，分割并转换为整数
        id = int(sp[2])  # 从 sp 的第三个元素中提取 ID，转换为整数
        cycle = int(sp[4])  # 从 sp 的第五个元素中提取 cycle 值，转换为整数

        return {'sys': sys, 'id': id, 'cycle': cycle}  # 返回一个字典，包含系统 ID、ID 和 cycle 值
    return  # 如果 'cycles' 不在 output 中，则返回 None
