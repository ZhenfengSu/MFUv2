import torch
import torch_tensorrt
import torchvision
model = torchvision.models.resnet18(pretrained=True).eval().cuda() # define your model here
x = torch.randn((1, 3, 224, 224)).cuda() # define what the inputs to the model will look like

optimized_model = torch.compile(model, backend="tensorrt")
optimized_model(x) # compiled on first run

optimized_model(x) # this will be fast!