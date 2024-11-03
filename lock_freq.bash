#!/bin/bash  

# 设置GPU频率限制范围 
sudo nvidia-smi -pm 1 
sudo nvidia-smi -lgc 2520

# 输出文件路径  
output_file="./txt/clock_info.txt"  

# 创建输出文件夹，如果不存在  
mkdir -p ./txt  

# 无限循环  
while true; do  
    # 1. 读取GPU0的频率  
    gpu_freq=$(nvidia-smi -i 0 --query-gpu=clocks.gr --format=csv,noheader,nounits)  
    
    # 2. 记录当前时间，格式为xxx年xx月xx日 xx时xx分xx秒  
    current_time=$(date "+%Y年%m月%d日 %H时%M分%S秒")  

    # 3. 写入到txt文件中  
    echo "$current_time GPU0 frequency: $gpu_freq MHz" >> "$output_file"  

    # 打印到终端（可选）  
    echo "$current_time GPU0 frequency: $gpu_freq MHz"  
    
    # 4. 等待1s  
    sleep 1  
done