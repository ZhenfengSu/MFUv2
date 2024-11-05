import torch
import torchvision
import os
import sys
# 获取当前的路径的目录
current_path = os.path.dirname(os.path.abspath(__file__))
# deit model
deit_model_path = os.path.join(current_path, 'deit_model')
sys.path.append(deit_model_path)
from deit_entiremodel import deit_base_patch16_LS
# vision llama
vision_llama_model_path = os.path.join(current_path, 'vision_llama')
sys.path.append(vision_llama_model_path)
from my_models import vanillaformer_large_patch16
# vanillanet
vanillanet_model_path = os.path.join(current_path, 'vanillanet_models')
sys.path.append(vanillanet_model_path)
from vanillanet import vanillanet_7
def create_model(model_type):
    if model_type == 'resnet18':
        model = torchvision.models.resnet18()
    elif model_type == 'resnet34':
        model = torchvision.models.resnet34()
    elif model_type == 'resnet50':
        model = torchvision.models.resnet50()
    elif model_type == 'deit_base':
        model = deit_base_patch16_LS()
    elif model_type == 'vanillaformer':
        model = vanillaformer_large_patch16()
    elif model_type == 'vanillanet':
        model = vanillanet_7()
    return model

if __name__ == '__main__':
    model = create_model('resnet18')
    print(model)