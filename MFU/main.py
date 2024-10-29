from functools import partial
from torch import nn as nn
from deit_entiremodel import vit_models, Layer_scale_init_Block
import argparse
from get_model import get_deit_model_info_map
from bs_search import binary_search_max_bs
import os
# 定义命令行参数
parser = argparse.ArgumentParser(description='MFU')
parser.add_argument('--hidden_dim', type=int, default=None, help='hidden_dim')
parser.add_argument('--depth', type=int, default=None, help='depth')
parser.add_argument('--flops', type=float, default=None, help='flops, unit is GMACs')
parser.add_argument('--params', type=float, default=None, help='params, unit is M')
parser.add_argument('--output', type=str, default='output_info.txt', help='output')
args = parser.parse_args()

# check output dir
if not os.path.exists('output'):
    os.makedirs('output')

# 获取模型信息
flops = args.flops
if flops is not None:
    flops = flops * 1e9 # GMACs to MACs
params = args.params
if params is not None:
    params = params * 1e6 # M to params
model_info_list = get_deit_model_info_map(args.hidden_dim, args.depth, flops, params)
print('get model info list')
# 遍历所有的模型信息
for model_info_map in model_info_list:
    # 创建模型
    hidden_dim = model_info_map["hidden_dim"]
    patch_size = model_info_map["patch_size"]
    heads = model_info_map["heads"]
    num_classes = model_info_map["num_classes"]
    depth = int(model_info_map["depth"])
    img_size = 224
    model = vit_models(
            img_size = img_size, patch_size=patch_size, embed_dim=hidden_dim, depth=depth, num_heads=heads, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block)
    # 计算最大的batch_size
    print("get max_bs for hidden_dim: " + str(hidden_dim) + " patch_size: " + str(patch_size) + " heads: " + str(heads) + " num_classes: " + str(num_classes) + " depth: " + str(depth))
    max_mfu, max_bs = binary_search_max_bs(model=model, model_info_map=model_info_map)
    # 输出结果
    print("hidden_dim: " + str(hidden_dim) + " patch_size: " + str(patch_size) + " heads: " + str(heads) + " num_classes: " + str(num_classes) + " depth: " + str(depth) + " max_mfu: " + str(max_mfu) + " max_bs: " + str(max_bs))
    with open('output/' + args.output, 'a') as f:
        f.write("hidden_dim: " + str(hidden_dim) + " patch_size: " + str(patch_size) + " heads: " + str(heads) + " num_classes: " + str(num_classes) + " depth: " + str(depth) + " max_mfu: " + str(max_mfu) + " max_bs: " + str(max_bs) + "\n")