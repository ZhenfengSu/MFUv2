'''
模拟transformer中的qkv的矩阵运算(其中一个)
'''
import torch
import argparse
import os
from torch.utils import benchmark
import time
batch_size_choices = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,1024,2048,4096,8192,16384]
def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")
    parser.add_argument("--shape_row", type=int, default=4096, help='shape_col')
    parser.add_argument('--output', type=str, default='output_gemm.txt', help='output')
    parser.add_argument("--gpu_type", type=str, default='RTX4090', help='gpu_type')
    parser.add_argument("--data_type", type=str, default='FP32', help='data_type')
    parser.add_argument("--shape_col", type=int, default=4096, help='shape_col')
    parser.add_argument("--device", type=str, default='cuda', help='device')
    parser.add_argument("--plot_mode", type=bool, default=False, help='plot_mode')
    args = parser.parse_args()
    return args

'''data type转换相关的代码'''
GPU_FLOPS_MAP = {
    'RTX4090': {'FP32': 82.6, 'FP16': 165.2,'INT8': 660.6},
    'RTX3080': {},
}
DATA_TYPE2TORCH = {
    'FP32': torch.float32,
    'FP16': torch.float16,
    'INT8': torch.int8,
}

def get_gpu_FLOPS(gpu_type, data_type):
    return GPU_FLOPS_MAP[gpu_type][data_type]

def geeem_sim(batch_size, data_type, device, data_shape_row, data_shape_col):
    typ = DATA_TYPE2TORCH[data_type]
    if data_type == 'FP32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data_shape_col = int(data_shape_col)
    data_shape_row = int(data_shape_row)
    a = torch.randn(batch_size, data_shape_row).type(typ).to(device)
    b = torch.randn(data_shape_row, data_shape_col).type(typ).to(device)
    
    t = benchmark.Timer(
        stmt='a @ b',
        globals={'a': a, 'b': b})
    # 改为 torch.mul(a, b) 以便测试
    # t = benchmark.Timer(
    #     stmt='torch.mm(a, b)',
    #     globals={'a': a, 'b': b})

    x = t.timeit(50)
    
    avg_time = x.median
    gemm_FLOPs = 2 * batch_size * data_shape_row * data_shape_col
    gemm_FLOPs_t = gemm_FLOPs / 1e12
    GPU_FLOPS = get_gpu_FLOPS('RTX4090', data_type)
    MFU = (gemm_FLOPs_t / GPU_FLOPS ) / avg_time
    FLOPS_shape = MFU * GPU_FLOPS
    # 获取当前的时间，以 xxx年xx月xx日xx时xx分xx秒 的格式表示
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("time: ", time_now)
    print("batch_size: ", batch_size)
    print("MFU: ", MFU)
    print("FLOPS_shape: ", FLOPS_shape)
    print("data_shape: ", str(data_shape_row)+'x'+str(data_shape_col))
    time.sleep(1)
    
    return  MFU, FLOPS_shape

def main():
    args = get_args_parser()
    device = torch.device(args.device)
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('./output/' + args.output, 'w') as f:
        f.write("\n")
        f.write("shape: " + str(args.shape_row)+'x'+str(args.shape_col) + "\n")
        f.write("\n")
    print("start")
    print("shape: " + str(args.shape_row)+'x'+str(args.shape_col))
    for batch_size in batch_size_choices:
        print("evaluating MFU and throughput")
        print("batch_size: " + str(batch_size))
        # 计算MFU和吞吐率
        MFU, FLOPS_shape = geeem_sim(batch_size, args.data_type, device, args.shape_row, args.shape_col)
        with open('./output/' + args.output, 'a') as f:
            if args.plot_mode:
                f.write("batch_size: " + str(batch_size) + " MFU: " + str(MFU)  + " FLOPS: " + str(FLOPS_shape) + "\n")
            else:
                f.write("\n")
                f.write("*"*30 + "\n")
                f.write("batch_size: " + str(batch_size) + "\n")
                f.write("MFU: " + str(MFU) + "\n")
                f.write("FLOPS_shape: " + str(FLOPS_shape) + "\n")
                f.write("*"*30 + "\n")
                f.write("\n")   
    print("done")

if __name__ == '__main__':
    main()