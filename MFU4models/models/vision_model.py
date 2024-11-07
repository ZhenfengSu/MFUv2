import torch
import torchvision
import os
import sys
# 获取当前的路径的目录
current_path = os.path.dirname(os.path.abspath(__file__))
# deit model
deit_model_path = os.path.join(current_path, 'deit_model')
sys.path.append(deit_model_path)
from deit_entiremodel import deit_tiny_patch16_LS,deit_small_patch16_LS,deit_medium_patch16_LS,deit_base_patch16_LS,deit_large_patch16_LS,deit_huge_patch14_LS
# vision llama
vision_llama_model_path = os.path.join(current_path, 'vision_llama')
sys.path.append(vision_llama_model_path)
from my_models import vanillaformer_large_patch16
# vanillanet
vanillanet_model_path = os.path.join(current_path, 'vanillanet_models')
sys.path.append(vanillanet_model_path)
from vanillanet import vanillanet_5,vanillanet_6, vanillanet_7, vanillanet_8, vanillanet_9, vanillanet_10,vanillanet_11,vanillanet_12,vanillanet_13,vanillanet_13_x1_5,vanillanet_13_x1_5_ada_pool

resnet_map = {
    'resnet18': torchvision.models.resnet18(),
    'resnet34': torchvision.models.resnet34(),
    'resnet50': torchvision.models.resnet50(),
    'resnet101': torchvision.models.resnet101(),
    'resnet152': torchvision.models.resnet152(),
}

deit_map = {
    'deit_tiny': deit_tiny_patch16_LS(),
    'deit_small': deit_small_patch16_LS(),
    'deit_medium': deit_medium_patch16_LS(),
    'deit_base': deit_base_patch16_LS(),
    'deit_large': deit_large_patch16_LS(),
    'deit_huge': deit_huge_patch14_LS(),
}

vanillanet_map = {
    'vanillanet_5': vanillanet_5(),
    'vanillanet_6': vanillanet_6(),
    'vanillanet_7': vanillanet_7(),
    'vanillanet_8': vanillanet_8(),
    'vanillanet_9': vanillanet_9(),
    'vanillanet_10': vanillanet_10(),
    'vanillanet_11': vanillanet_11(),
    'vanillanet_12': vanillanet_12(),
    'vanillanet_13': vanillanet_13(),
    'vanillanet_13_x1_5': vanillanet_13_x1_5(),
    'vanillanet_13_x1_5_ada_pool': vanillanet_13_x1_5_ada_pool(),
}

vgg_map = {
    'vgg11': torchvision.models.vgg11(),
    'vgg11_bn': torchvision.models.vgg11_bn(),
    'vgg13': torchvision.models.vgg13(),
    'vgg13_bn': torchvision.models.vgg13_bn(),
    'vgg16': torchvision.models.vgg16(),
    'vgg16_bn': torchvision.models.vgg16_bn(),
    'vgg19': torchvision.models.vgg19(),
    'vgg19_bn': torchvision.models.vgg19_bn(),
}

def create_model(model_type):
    if model_type.startswith('resnet'):
        return resnet_map[model_type]
    elif model_type.startswith('deit'):
        return deit_map[model_type]
    elif model_type.startswith('vanillanet'):
        return vanillanet_map[model_type]
    elif model_type.startswith('vgg'):
        return vgg_map[model_type]
    else:
        raise ValueError('model_type not supported')
    return model
def get_params(model):
    return sum([param.nelement() for param in model.parameters()]) / 1e6
'''
small model
resnet34 : 21.797672
deit_small : 22.059496
vanillanet_5 : 22.3288

'''
'''
medium model
resnet152 : 60.192808
deit_medium : 38.849512
vanillanet_6 : 56.116704

'''
'''
large model
vanillanet_13_x1_5 : 236.942048
deit_large : 304.37476
mine

'''


'''
resnet18 : 11.689512
resnet34 : 21.797672
resnet50 : 25.557032
resnet101 : 44.54916
resnet152 : 60.192808

deit_tiny : 5.721832
deit_small : 22.059496
deit_medium : 38.849512
deit_base : 86.58532
deit_large : 304.37476
deit_huge : 632.12644

vanillanet_5 : 22.3288
vanillanet_6 : 56.116704
vanillanet_7 : 56.670176
vanillanet_8 : 65.17552
vanillanet_9 : 73.680864
vanillanet_10 : 82.186208
vanillanet_11 : 90.691552
vanillanet_12 : 99.196896
vanillanet_13 : 107.70224
vanillanet_13_x1_5 : 236.942048
vanillanet_13_x1_5_ada_pool : 236.942048

vgg11 : 132.863336
vgg11_bn : 132.86884
vgg13 : 133.047848
vgg13_bn : 133.053736
vgg16 : 138.357544
vgg16_bn : 138.365992
vgg19 : 143.66724
vgg19_bn : 143.678248
'''
if __name__ == '__main__':
    # model = create_model('resnet101')
    # print(model)
    # # 获得模型的参数量
    # print(get_params(model)) 
    # 测试所有模型的参数量，并打印出来
    for model_name in resnet_map:
        model = create_model(model_name)
        print(model_name, ':',get_params(model))
        
    for model_name in deit_map:
        model = create_model(model_name)
        print(model_name, ':',get_params(model))
        
    for model_name in vanillanet_map:
        model = create_model(model_name)
        print(model_name, ':',get_params(model))
        
    for model_name in vgg_map:
        model = create_model(model_name)
        print(model_name, ':',get_params(model))