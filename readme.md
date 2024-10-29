# MFU

测试某一个模型的MFU和THROGUHPUTd的代码在MFU4models文件夹下

命令示例：
```bash

cd MFU4models
# GPU: 4090
# resnet 18
python main.py --model-type resnet18 --plot_mode True --output resnet18_info.txt --plot_mode True # FP32
python main.py --model-type resnet18 --plot_mode True --output resnet18_fp16_info.txt --plot_mode True --data_type FP16 # FP16
python main.py --model-type resnet18 --plot_mode True --output resnet18_int8_info.txt --plot_mode True --data_type INT8 # INT8
# ddp launch
python -m torch.distributed.launch --nproc_per_node 8 main_ddp.py --model-type resnet18 --plot_mode True --output resnet18_ddp_info.txt --plot_mode True # FP32
# ddp spawn
python main_ddp_spawn.py --plot_mode True --output resnet18_ddp_info.txt --plot_mode True # FP32
python main_ddp_spawn.py --plot_mode True --output output_resnet.txt
```

其中单卡正常
采用ddp launch的多卡测试非常慢，而且batch_size越大跑越慢
采用ddp spawn的多卡测试速度正常，但是MFU测试数据异常大，甚至会出现大于一的情况（理论上完全不可能！）