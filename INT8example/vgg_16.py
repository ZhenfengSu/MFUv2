import torch
import torchvision

model = torchvision.models.vgg16(pretrained=True,download=True)
model.eval()
model = model.cuda()
