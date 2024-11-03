# deit base fix freq fp32
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_deit/deit_base_info.txt
#max MFU:  0.3749934455270558
# max batch_size:  64
# min MFU:  0.17029415742004386
# min batch_size:  1

# deit base fix freq fp16
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_deit/deit_base_fp16_info.txt
# max MFU:  0.6175236110477162
# max batch_size:  64
# min MFU:  0.14875078343600118
# min batch_size:  1

# deit base fix freq int8
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_deit/deit_base_int8_info.txt
# max MFU:  0.07413812569888263
# max batch_size:  16
# min MFU:  0.0327041593051824
# min batch_size:  1

# deit base vary freq fp32
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_deit/deit_base_info.txt
# max MFU:  0.3690614050025595
# max batch_size:  256
# min MFU:  0.1654944919667008
# min batch_size:  1

# deit base vary freq fp16
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_deit/deit_base_fp16_info.txt
# max MFU:  0.6325430338389709
# max batch_size:  64
# min MFU:  0.14886426728753907
# min batch_size:  1

# deit base vary freq int8
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_deit/deit_base_int8_info.txt
# max MFU:  0.07688496894150397
# max batch_size:  16
# min MFU:  0.032313320873818545
# min batch_size:  1

# resnet18 fix freq fp32
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_resnet18/resnet18_info.txt
# max MFU:  0.38445861514032326
# max batch_size:  32
# min MFU:  0.08011568340367001
# min batch_size:  1

# resnet18 fix freq fp16
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_resnet18/resnet18_fp16_info.txt
# max MFU:  0.3109877464515373
# max batch_size:  128
# min MFU:  0.033239721190362226
# min batch_size:  1

# resnet18 fix freq int8
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_fix_freq_resnet18/resnet18_int8_info.txt
# max MFU:  0.02002611335391624
# max batch_size:  64
# min MFU:  0.0030732045403559127
# min batch_size:  1

# resnet18 vary freq fp32
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_resnet18/resnet18_info.txt

# max MFU:  0.41371274261252217
# max batch_size:  32
# min MFU:  0.06642315261045345
# min batch_size:  1

# resnet18 vary freq fp16
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_resnet18/resnet18_fp16_info.txt
# max MFU:  0.35162846230960054
# max batch_size:  64
# min MFU:  0.034317986627498596
# min batch_size:  1

# resnet18 vary freq int8
python model_result.py --txt_path /home/smi/rank_project/MFUv2/MFU4models/output_vary_freq_resnet18/resnet18_int8_info.txt
# max MFU:  0.0197775672139543
# max batch_size:  16
# min MFU:  0.003014010297788997
# min batch_size:  1