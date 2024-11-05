from models.vision_model import create_model
import argparse
from flops_count_model import flops_count, mfu_throughput_FLOPS, get_model_quantization, DATA_TYPE2TORCH
import os
import torch
import torchvision
import time
import torchvision.transforms as transforms
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
batch_size_choice = [1,2,4,8,16,32,64,128,256]
def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--model-type", type=str, default='resnet18', help='model_type')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--output', type=str, default='output_resnet.txt', help='output')
    parser.add_argument("--gpu_type", type=str, default='RTX4090', help='gpu_type')
    parser.add_argument("--data_type", type=str, default='FP32', help='data_type')
    parser.add_argument("--device", type=str, default='cuda', help='device')
    parser.add_argument("--plot_mode", type=bool, default=False, help='plot_mode')
    args = parser.parse_args()
    return args

def main(args):
    device = torch.device(args.device)
    if not os.path.exists('output_ddp'):
        os.makedirs('output_ddp')
    model_flops = create_model(args.model_type)
    model_inference = create_model(args.model_type)
    input_dtype = DATA_TYPE2TORCH[args.data_type]
    model_inference = get_model_quantization(model_inference, args.data_type)
    print(model_flops)
    # 测试flops和params(一次推理，batch_size=1,单位为MACs)
    FLOPs, Params = flops_count(model_flops)
    
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device("cuda", local_rank)
    model_inference = model_inference.to(device)
    model_inference_ddp = DDP(model_inference, device_ids=[local_rank], output_device=local_rank)
    # 定义转换序列，将图像调整为 224x224 并转换为张量  
    transform = transforms.Compose([  
        transforms.Resize((224, 224)), # 调整图像大小为 224x224  
        transforms.ToTensor(),         # 转换为张量  
    ])  
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    if dist.get_rank() == 0:
        with open('./output_ddp/' + args.output, 'w') as f:
            f.write("\n")
            f.write("model_type: " + args.model_type + "\n")
            f.write("\n")
    for batch_size in batch_size_choice:
        print("evaluating MFU and throughput")
        print("batch_size: " + str(batch_size))
        # 模拟推理的时间测试
        train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=train_sampler)
        trainloader.sampler.set_epoch(0)
        time_list = []
        for i , (data, label) in enumerate(trainloader):
            data = data.to(device)
            data = data.to(input_dtype)
            time_begin = time.time()
            prediction = model_inference_ddp(data)
            time_end = time.time()
            time_list.append(time_end - time_begin)
            torch.cuda.synchronize()
            if i >= 40:
                break
        time_list = time_list[10:] # 抛弃前10次
        # 计算MFU和吞吐率
        MFU, THROUGHPUT, FLOPS = mfu_throughput_FLOPS(FLOPs, batch_size, time_list, args.data_type, args.gpu_type)
        if dist.get_rank() == 0:
            with open('./output_ddp/' + args.output, 'a') as f:
                if args.plot_mode:
                    f.write("batch_size: " + str(batch_size) + " MFU: " + str(MFU) + " THROUGHPUT: " + str(THROUGHPUT) + " FLOPS: " + str(FLOPS) + "\n")
                    f.write("time_list: " + str(time_list) + "\n")
                else:
                    f.write("\n")
                    f.write("*"*30 + "\n")
                    f.write("batch_size: " + str(batch_size) + "\n")
                    f.write("FLOPs: " + str(FLOPs) + "\n")
                    f.write("Params: " + str(Params) + "\n")
                    f.write("MFU: " + str(MFU) + "\n")
                    f.write("THROUGHPUT: " + str(THROUGHPUT) + "\n")
                    f.write("FLOPS_model: " + str(FLOPS) + "\n")
                    f.write("time_list: " + str(time_list) + "\n")
                    f.write("*"*30 + "\n")
                    f.write("\n")
    print("done")
    



if __name__ == '__main__':
    args = get_args_parser()
    main(args)