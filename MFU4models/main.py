from vision_model import create_model
import argparse
from flops_count_model import flops_count, model_inference_sim, mfu_throughput_FLOPS
import os
import torch
batch_size_choice = [1,2,4,8,16,32,64,128,256,512,1024]
def get_args_parser():
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")
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
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('./output/' + args.output, 'w') as f:
        f.write("\n")
        f.write("model_type: " + args.model_type + "\n")
        f.write("\n")
    model_flops = create_model(args.model_type)
    model_inference = create_model(args.model_type)
    print(model_flops)
    # 测试flops和params(一次推理，batch_size=1,单位为MACs)
    FLOPs, Params = flops_count(model_flops)
    for batch_size in batch_size_choice:
        try:
            print("evaluating MFU and throughput")
            print("batch_size: " + str(batch_size))
            # 模拟推理的时间测试
            time_list = model_inference_sim(model_inference, batch_size, args.data_type, device)
            # 计算MFU和吞吐率
            MFU, THROUGHPUT, FLOPS = mfu_throughput_FLOPS(FLOPs, batch_size, time_list, args.data_type, args.gpu_type)
            with open('./output/' + args.output, 'a') as f:
                if args.plot_mode:
                    f.write("batch_size: " + str(batch_size) + " MFU: " + str(MFU) + " THROUGHPUT: " + str(THROUGHPUT) + " FLOPS: " + str(FLOPS) + "\n")
                    # f.write("time_list: " + str(time_list) + "\n")
                else:
                    f.write("\n")
                    f.write("*"*30 + "\n")
                    f.write("batch_size: " + str(batch_size) + "\n")
                    f.write("FLOPs: " + str(FLOPs) + "\n")
                    f.write("Params: " + str(Params) + "\n")
                    f.write("MFU: " + str(MFU) + "\n")
                    f.write("THROUGHPUT: " + str(THROUGHPUT) + "\n")
                    f.write("FLOPS_model: " + str(FLOPS) + "\n")
                    # f.write("time_list: " + str(time_list) + "\n")
                    f.write("*"*30 + "\n")
                    f.write("\n")
        except Exception as e:
            print(e)
    print("done")
    



if __name__ == '__main__':
    args = get_args_parser()
    main(args)