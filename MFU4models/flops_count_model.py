from flops_counter import get_model_complexity_info
import torch
import time
'''data type转换相关的代码'''
GPU_FLOPS_MAP = {
    'RTX4090': {'FP32': 82.6, 'FP16': 330.3,'INT8': 660.6},
    'RTX3080': {},
}
DATA_TYPE2TORCH = {
    'FP32': torch.float32,
    'FP16': torch.float16,
    'INT8': torch.int8,
}

def get_gpu_FLOPS(gpu_type, data_type):
    return GPU_FLOPS_MAP[gpu_type][data_type]

def get_model_quantization(model, data_type):
    if data_type == 'FP32':
        return model
    elif data_type == 'FP16':
        model = model.half()
    elif data_type == 'INT8':
        model = model.int8()
    return model

'''计算FLOPs相关的代码'''
def flops_count(model):
    # build input
    input = (3,224,224)
    custom_modules_hooks = {}
    # flops
    flops, params = get_model_complexity_info(model, input, custom_modules_hooks=custom_modules_hooks,as_strings=False)
    print(flops, params)   
    return flops, params
    
'''单卡推理的代码'''
def model_inference_sim(model, batch_size,data_type=None, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build input
    if data_type is None:
        data_type = 'FP32'
    input = torch.randn(batch_size, 3, 224, 224, dtype=DATA_TYPE2TORCH[data_type])
    model = get_model_quantization(model, data_type)
    # inference
    model.eval()
    time_list = []
    # put model to device
    model = model.to(device)
    input = input.to(device)
    # forward
    for i in range(40):
        time_begin = time.time()
        input = input.to(device)
        output = model(input)
        torch.cuda.synchronize()   
        time_end = time.time() 
        time_list.append(time_end - time_begin)
    # 抛弃第1次
    time_list = time_list[1:]
    # 把模型放回CPU
    model = model.to("cpu")
    input = input.to("cpu")
    # 清空显存
    torch.cuda.empty_cache()
    return time_list
    

'''计算MFU和吞吐率'''
def mfu_throughput_FLOPS(FLOPs, batch_size, time_list, data_type, gpu_type):
    # 把batch_size为1的前向推理的FLOPs转换成batch_size的FLOPs
    FLOPs_bs = FLOPs * batch_size
    # 先转换单位
    FLOPs_bs_T = 2*FLOPs_bs / (1e12)
    # 计算吞吐率
    data_size = batch_size * len(time_list)
    THROUGHPUT = data_size / sum(time_list)
    # 获取显卡的性能
    FLOPS_GPU = get_gpu_FLOPS(gpu_type, data_type)
    # 计算MFU
    MFU = FLOPs_bs_T * THROUGHPUT / (FLOPS_GPU * batch_size)
    # 计算模型的FLOPS
    FLOPS = MFU * FLOPS_GPU
    return MFU, THROUGHPUT, FLOPS