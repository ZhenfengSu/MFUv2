# fix_freq fp32
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_fix_freq/output_fp32
# 16384x8192x8192:0.9852354003310523
# 128x512x512:0.11620945563401713

# fix_freq fp16
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_fix_freq/output_fp16
# 4096x2560x2560:0.961019934592907
# 128x512x512:0.07318670282922762

# fix_freq int8
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_fix_freq/output_of16 --data_type INT8
# 1024x4096x4096:0.8018227891478882
# 128x512x512:0.015548890433928191
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_fix_freq/output_oint32 --data_type INT8
# 1024x4096x4096:0.7568981193140649
# 128x512x512:0.015751483487879567

# vary_freq fp32
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_vary_freq/output_fp32
# 4096x8192x8192:1.0292352986593958
# 128x512x512:0.11650152150317289

# vary_freq fp16
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_vary_freq/output_fp16
# 4096x2560x2560:0.9958087150902902
# 128x512x512:0.0722957847654766

# vary_freq int8
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_vary_freq/output_of16 --data_type INT8
# 4096x3072x3072:0.7989198621605332
# 128x512x512:0.01665940934916878
python gemm_result.py --txt_path /home/smi/rank_project/GEMM/output_vary_freq/output_oint32 --data_type INT8
# 512x3712x3712:0.7242691200298426
# 128x512x512:0.013857631586435519