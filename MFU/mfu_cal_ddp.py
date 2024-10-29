'''
计算多卡的前向推理MFU和吞吐量
'''
import os
import torch
from get_model import flops_deit
import time
# model
import torch.nn as nn
from get_model import get_deit_model_info_map, flops_deit
from deit_entiremodel import vit_models, Layer_scale_init_Block
from functools import partial
# 新增1:依赖
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
FP32_4090 = 82.6 # 单位是TFLOPS

# 模拟多卡的数据
def SimDataset(batch_size, num_devices):
    # simulate data for 40 iterations
    total_size = batch_size * num_devices * 100
    data = torch.randn(total_size, 3, 224, 224)
    label = torch.randint(0, 1000, (total_size,))
    return list(zip(data, label))



def mfu_throughput_ddp(args, model_info_map, num_devices, time_list):
    # 总的数据量
    steps = len(time_list)
    print("steps: " + str(steps))
    print("batch_size: " + str(args.batch_size))
    print("num_devices: " + str(num_devices))
    print("total time: " + str(sum(time_list)))
    print("time_list: " + str(time_list))
    
    data_size = args.batch_size * num_devices * steps
    # 计算吞吐率
    throughput = data_size / (sum(time_list)*num_devices)
    
    # 计算一次推理的FLOPS
    hidden_dim = model_info_map["hidden_dim"]
    patch_size = model_info_map["patch_size"]
    num_classes = 1000
    depth = model_info_map["depth"]
    # 计算flops(flops_deit是bs=1的flops)
    flops_forward = flops_deit(hidden_dim, patch_size, num_classes, depth)* args.batch_size
    # flops_total的单位是MACs,所以要转换成TFLOPS(乘以2然后除以1e12)
    flops_total = 2*flops_forward / (1e12)
    
    # 计算MFU
    MFU = flops_total * throughput /(FP32_4090 * args.batch_size)
    
    return MFU, throughput

def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden_dim')
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--output', type=str, default='output_ddp_info.txt', help='output')
    args = parser.parse_args()
    return args

def main():
    args = get_args_parser()
    
    # DDP backend初始化
    # a.根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(args.local_rank)
    # b.初始化DDP，使用默认backend(nccl)就行
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device("cuda", args.local_rank)
    
    # 获取模型信息
    model_info_list = get_deit_model_info_map(args.hidden_dim, args.depth, None, None)
    model_info_map = model_info_list[0]
    
    # 创建模型
    print("create model")
    hidden_dim = model_info_map["hidden_dim"]
    patch_size = model_info_map["patch_size"]
    heads = model_info_map["heads"]
    num_classes = model_info_map["num_classes"]
    depth = int(model_info_map["depth"])
    img_size = 224
    model = vit_models(
            img_size = img_size, patch_size=patch_size, embed_dim=hidden_dim, depth=depth, num_heads=heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block)
    # 将模型加载到对应的GPU上
    model = model.to(device)
    
    # 初始化DDP模型
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # 获取数据
    num_gpus = dist.get_world_size()
    print("num_gpus: " + str(num_gpus))
    my_train_data = SimDataset(args.batch_size, num_gpus)
    breakpoint()
    # 新增1：使用DistributedSampler，DDP帮我们把细节都封装起来了。用，就完事儿！
    # sampler的原理，后面也会介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_train_data)
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_train_data, batch_size=args.batch_size, sampler=train_sampler)
    time_list = []
    
    for i , (data , label) in enumerate(trainloader):
        local_rank = dist.get_rank()
        print("iter: " + str(i), "local_rank: " + str(local_rank))
        time_begin = time.time()
        output = model(data) 
        time_end = time.time() 
        time_list.append(time_end - time_begin)
    
    if dist.get_rank() == 0:
        # 计算多卡吞吐率和MFU
        time_list = time_list[50:] # 抛弃前50次
        MFU, throughput = mfu_throughput_ddp(args, model_info_map, num_gpus, time_list)
        
        if not os.path.exists('output'):
            os.makedirs('output')
        with open('./output/'+ args.output, 'a') as f:
            f.write('\n')
            f.write('*'*30 + '\n')
            f.write("model info"+ '\n')
            f.write("hidden_dim: " + str(hidden_dim) + '\n')
            f.write("patch_size: " + str(patch_size) + '\n')
            f.write("heads: " + str(heads) + '\n')
            f.write("num_classes: " + str(num_classes) + '\n')
            f.write("depth: " + str(depth) + '\n')
            f.write("inference time"+ '\n')
            f.write(str(time_list) + '\n')
            f.write("mfu and throughput"+ '\n')
            f.write("mfu: " + str(MFU) + '\n')
            f.write("throughput: " + str(throughput) + '\n')
            f.write('*'*30 + '\n')
            f.write('\n')
    return

if __name__ == '__main__':
    main()
        