# python gemm_eval.py --shape_row 4096 --shape_col 4096 --output shape_4096_info.txt
# python gemm_eval.py --shape_row 4096 --shape_col 4096 --data_type FP16 --plot_mode True --output shape_4096_fp16_info.txt
# # 8192 方阵
# python gemm_eval.py --shape_row 8192 --shape_col 8192 --plot_mode True --output shape_8192_fp32_info.txt
# python gemm_eval.py --shape_row 4096 --shape_col 16384 --plot_mode True --output shape_4096_16384_fp32_info.txt
# python gemm_eval.py --shape_row 16384 --shape_col 4096 --plot_mode True --output shape_16384_4096_fp32_info.txt

log_file="./txt/output.txt"
python gemm_eval.py --shape_row 512 --shape_col 512 --plot_mode True --output shape_512_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 768 --shape_col 768 --plot_mode True --output shape_768_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 1024 --shape_col 1024 --plot_mode True --output shape_1024_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 1280 --shape_col 1280 --plot_mode True --output shape_1280_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 1664 --shape_col 1664 --plot_mode True --output shape_1664_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 1792 --shape_col 1792 --plot_mode True --output shape_1792_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 1920 --shape_col 1920 --plot_mode True --output shape_1920_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 2048 --shape_col 2048 --plot_mode True --output shape_2048_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 2240 --shape_col 2240 --plot_mode True --output shape_2240_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 2432 --shape_col 2432 --plot_mode True --output shape_2432_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 2560 --shape_col 2560 --plot_mode True --output shape_2560_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 2816 --shape_col 2816 --plot_mode True --output shape_2816_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 3072 --shape_col 3072 --plot_mode True --output shape_3072_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 3328 --shape_col 3328 --plot_mode True --output shape_3328_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 3712 --shape_col 3712 --plot_mode True --output shape_3712_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 3840 --shape_col 3840 --plot_mode True --output shape_3840_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 3968 --shape_col 3968 --plot_mode True --output shape_3968_fp32_info.txt >> $log_file
python gemm_eval.py --shape_row 4096 --shape_col 4096 --plot_mode True --output shape_4096_fp32_info.txt >> $log_file


# int8
python gemm_eval.py --shape_row 512 --shape_col 512 --plot_mode True --output shape_512_int8_info.txt --data_type INT8