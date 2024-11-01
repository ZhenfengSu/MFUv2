"""
.. _vgg16_ptq:

Deploy Quantized Models using Torch-TensorRT
======================================================

Here we demonstrate how to deploy a model quantized to INT8 or FP8 using the Dynamo frontend of Torch-TensorRT
"""

# %%
# Imports and Model Definition

import argparse

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt as torchtrt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from modelopt.torch.quantization.utils import export_torch_mode

PARSER = argparse.ArgumentParser(
    description="Load pre-trained VGG model and then tune with FP8 and PTQ. For having a pre-trained VGG model, please refer to https://github.com/pytorch/TensorRT/tree/main/examples/int8/training/vgg16"
)
PARSER.add_argument(
    "--batch-size",
    default=128,
    type=int,
    help="Batch size for tuning the model with PTQ and FP8",
)
args = PARSER.parse_args()

model = torchvision.models.resnet18(pretrained=True)
model = model.cuda()

model.eval()

training_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = transforms.Compose([  
        transforms.Resize((224, 224)), # 调整图像大小为 224x224  
        transforms.ToTensor(),         # 转换为张量  
    ]),
)
training_dataloader = torch.utils.data.DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    drop_last=True,
)

data = iter(training_dataloader)
images, _ = next(data)


def calibrate_loop(model):
    # calibrate over the training dataset
    for data, labels in training_dataloader:
        data, labels = data.cuda(), labels.cuda(non_blocking=True)
        out = model(data)


# Tune the pre-trained model with FP8 and PTQ
quant_cfg = mtq.INT8_DEFAULT_CFG
# PTQ with in-place replacement to quantized modules
mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

# Inference
# Load the testing dataset
with torch.no_grad():
    with export_torch_mode():
        # Compile the model with Torch-TensorRT Dynamo backend
        input_tensor = images.cuda()
        # torch.export.export() failed due to RuntimeError: Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()
        from torch.export._trace import _export

        exp_program = _export(model, (input_tensor,))
        enabled_precisions = {torch.int8}
        trt_model = torchtrt.dynamo.compile(
            exp_program,
            inputs=[input_tensor],
            enabled_precisions=enabled_precisions,
            min_block_size=1,
            debug=False,
        )
        import time
        begin_time = time.time()
        batch_size = 128
        input_tensor = torch.randn(batch_size, 3, 224, 224).cuda()
        for i in range(100):
            output = trt_model(input_tensor)
        end_time = time.time()
        print("Inference Time: {:.5f}".format(end_time - begin_time))
        