import torch
from torch.utils import benchmark
import argparse
def get_args_parser():
    parser = argparse.ArgumentParser(description="GPU FLOPS")
    parser.add_argument("--data_type", type=str, default='FP32', help='data_type')
    parser.add_argument("--shape", type=str, default='128,128,128', help='shape')
    args = parser.parse_args()
    return args

def main(args):
    # 获取输入参数
    data_type = args.data_type
    shape = tuple(map(int, args.shape.split(',')))
    # 创建输入数据
    input_data = torch.randn(shape).to(data_type)
    # 创建模型
    model = torch.nn.Conv3d(3, 3, 3).to(data_type)
    # 进行推理
    f = benchmark.Timer(
        stmt='model(input_data)',
        globals={'model': model, 'input_data': input_data}
    ).blocked_autorange(min_run_time=1)
    print(f)
    
def main2(args):
    import torch  
    import time  

    # 确定我们是否可使用 GPU  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # 定义矩阵的大小  
    N = 8192  

    # 生成两个随机矩阵 A 和 B  
    A = torch.randn(N, N, device=device)  
    B = torch.randn(N, N, device=device)  

    # 执行矩阵乘法并测量时间  
    start_time = time.time()  
    C = torch.mm(A, B)  # 执行 A @ B  
    end_time = time.time()  

    # 打印运行时间  
    print(f"Matrix multiplication (GEMM) of size {N}x{N} completed in {end_time - start_time:.4f} seconds.")