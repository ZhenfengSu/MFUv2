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



# deit base
python main.py --model-type deit_base --plot_mode True --output deit_base_info.txt --plot_mode True # FP32
python main.py --model-type deit_base --plot_mode True --output deit_base_fp16_info.txt --plot_mode True --data_type FP16 # FP16
python main.py --model-type deit_base --plot_mode True --output deit_base_int8_info.txt --plot_mode True --data_type INT8 # INT8

python main.py --model-type vanillaformer --plot_mode True --output vanillaformer_info.txt --plot_mode True # FP32
python main.py --model-type vanillaformer --plot_mode True --output vanillaformer_fp16_info.txt --plot_mode True --data_type FP16 # FP16