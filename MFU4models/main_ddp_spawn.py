from models.vision_model import create_model
import argparse
from flops_count_model import flops_count, mfu_throughput_FLOPS, get_model_quantization, DATA_TYPE2TORCH
import os
import torch
import torchvision
import time
import torchvision.transforms as transforms
import torch.multiprocessing as mp
# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
batch_size_choice = [1,2,4,8,16,32,64,128,256,512,1024]
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

def main_ddp(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cuda_device = rank          
    args = get_args_parser()              
    if not os.path.exists('output_ddp'):
        os.makedirs('output_ddp')
    model_flops = create_model(args.model_type)
    model_inference = create_model(args.model_type)
    input_dtype = DATA_TYPE2TORCH[args.data_type]
    model_inference = get_model_quantization(model_inference, args.data_type)
    # 测试flops和params(一次推理，batch_size=1,单位为MACs)
    FLOPs, Params = flops_count(model_flops)

    # DDP Model
    model_inference = model_inference.to(cuda_device)
    model_inference = DDP(model_inference, device_ids=[rank])
    model_inference.eval()
    # 定义转换序列，将图像调整为 224x224 并转换为张量  
    transform = transforms.Compose([  
        transforms.Resize((224, 224)), # 调整图像大小为 224x224  
        transforms.ToTensor(),         # 转换为张量  
    ])  
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    if rank == 0:
        with open('./output_ddp/' + args.output, 'w') as f:
            f.write("\n")
            f.write("model_type: " + args.model_type + "\n")
            f.write("\n")
    for batch_size in batch_size_choice:
        print("evaluating MFU and throughput")
        print("batch_size: " + str(batch_size))
        # 模拟推理的时间测试
        val_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
        val_loader = torch.utils.data.DataLoader(my_trainset, batch_size=batch_size, sampler=val_sampler, shuffle=False, num_workers=1, pin_memory=True)
        val_loader.sampler.set_epoch(0)
        time_list = []
        with torch.no_grad():
            if rank == 0:
                for i , (data, label) in enumerate(val_loader):
                    time_begin = time.time()
                    if cuda_device is not None:
                        data = data.to(cuda_device, non_blocking=True)
                    data = data.to(input_dtype)
                    prediction = model_inference(data)
                    torch.cuda.synchronize()
                    if i >= 40:
                        break
                    time_end = time.time()
                    time_list.append(time_end - time_begin)
                # print("time: ", time_end - time_begin)
            else:
                for i , (data, label) in enumerate(val_loader):
                    if cuda_device is not None:
                        data = data.to(cuda_device, non_blocking=True)
                    data = data.to(input_dtype)
                    prediction = model_inference(data)
                    torch.cuda.synchronize()
                    if i >= 40:
                        break
        if rank == 0:
            time_list = time_list[1:] # 去掉第一个时间
            # 计算MFU和吞吐率
            MFU, THROUGHPUT, FLOPS = mfu_throughput_FLOPS(FLOPs, batch_size, time_list, args.data_type, args.gpu_type)
        if rank == 0:
            with open('./output_ddp/' + args.output, 'a') as f:
                if args.plot_mode:
                    f.write("batch_size: " + str(batch_size) + " MFU: " + str(MFU) + " THROUGHPUT: " + str(THROUGHPUT) + " FLOPS: " + str(FLOPS) + "\n")
                    f.write("time_list: " + str(time_list) + "\n")
                else:
                    f.write("\n")
                    f.write("*"*30 + "\n")
                    f.write("batch_size: " + str(batch_size) + "\n")
                    f.write("FLOPs: " + str(FLOPs) + "Macs \n")
                    f.write("Params: " + str(Params) + "\n")
                    f.write("MFU: " + str(MFU*100) + "%\n")
                    f.write("THROUGHPUT: " + str(THROUGHPUT) + "images/s\n")
                    f.write("FLOPS_model: " + str(FLOPS) + "TFLOPS\n")
                    f.write("time_list: " + str(time_list) + "\n")
                    f.write("time_avg: " + str(sum(time_list)/len(time_list)) + "seconds\n")
                    f.write("*"*30 + "\n")
                    f.write("\n")
    print("done")
    dist.destroy_process_group()



if __name__ == '__main__':
    world_size = 8
    mp.spawn(main_ddp, args=(world_size, ), nprocs=world_size, join=True)