from flops_counter import get_model_complexity_info
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.utils import export_torch_mode
import torch_tensorrt as torchtrt
from copy import deepcopy
'''data type转换相关的代码'''
GPU_FLOPS_MAP = {
    'RTX4090': {'FP32': 82.6, 'FP16': 165.2,'INT8': 660.6},
    'RTX3080': {},
}
DATA_TYPE2TORCH = {
    'FP32': torch.float32,
    'FP16': torch.float16,
    'INT8': torch.float32, # we use the tensorrt to do the INT8 quantization, so the data type is FP32
}

def get_gpu_FLOPS(gpu_type, data_type):
    return GPU_FLOPS_MAP[gpu_type][data_type]

def calibrate_dataloader_inputtensor(batch_size, device):
    training_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = transforms.Compose([  
        transforms.Resize((224, 224)), # 调整图像大小为 224x224  
        transforms.ToTensor(),         # 转换为张量  
    ]),
    )
    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    images, _ = next(iter(training_dataloader))
    images = images.to(device)
    return training_dataloader, images

def get_model_quantization(model, data_type, batch_size, device):
    model = model.to(device)
    if data_type == 'FP32':
        return model
    elif data_type == 'FP16':
        model = model.half()
        return model
    elif data_type == 'INT8':
        training_dataloader, input_tensor = calibrate_dataloader_inputtensor(batch_size, device)
        def calibrate_loop(model):
            # calibrate over the training dataset
            for i ,(data, labels) in enumerate(training_dataloader):
                data, labels = data.cuda(), labels.cuda(non_blocking=True)
                out = model(data)
                if i > 20:
                    break
        quant_cfg = mtq.INT8_DEFAULT_CFG
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
        with export_torch_mode():
            input_tensor = input_tensor.to(device)
            from torch.export._trace import _export
            
            exp_program = _export(model, (input_tensor,))
            enabled_precisions = {torch.int8}
            trt_model = torchtrt.dynamo.compile(
                exp_program,
                inputs=[input_tensor],
                enabled_precisions=enabled_precisions,
            )
        
        return trt_model


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
    inference_model = deepcopy(model)
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build input
    if data_type is None:
        data_type = 'FP32'
    if data_type == 'INT8':
        input = torch.randint(0, 2, (batch_size, 3, 224, 224), dtype=DATA_TYPE2TORCH[data_type])
    else:
        input = torch.randn(batch_size, 3, 224, 224, dtype=DATA_TYPE2TORCH[data_type])
        
    inference_model = get_model_quantization(inference_model, data_type, batch_size, device)
    # inference
    inference_model.eval()
    time_list = []
    # put model to device
    print("ready to inference")
    inference_model = inference_model.to(device)
    input = input.to(device)
    # forward
    with torch.no_grad():
        for i in range(40):
            time_begin = time.time()
            input = input.to(device)
            output = inference_model(input)
            torch.cuda.synchronize()   
            time_end = time.time() 
            time_list.append(time_end - time_begin)
    # 抛弃第1次
    time_list = time_list[1:]
    # 把模型放回CPU
    model = model.to("cpu")
    input = input.to("cpu")
    # 删除inference_model
    del inference_model
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