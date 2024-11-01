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

python main.py --model-type resnet34 --plot_mode True --output resnet34_info.txt --plot_mode True
python main.py --model-type resnet50 --plot_mode True --output resnet50_info.txt --plot_mode True



python main.py --model-type resnet18 --output resnet18_int8_info_eval1.txt --data_type INT8