import torch
import torchvision
from deit_entiremodel import deit_base_patch16_LS
def create_model(model_type):
    if model_type == 'resnet18':
        model = torchvision.models.resnet18()
    elif model_type == 'resnet34':
        model = torchvision.models.resnet34()
    elif model_type == 'resnet50':
        model = torchvision.models.resnet50()
    elif model_type == 'deit_base':
        model = deit_base_patch16_LS()
    return model