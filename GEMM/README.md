# 测试GEMM在不同精度下的性能

## 1. 测试环境
`CPU： Intel i9-13900K`

`GPU： RTX 4090`

`内存： 64GB`

`操作系统： Ubuntu 20.04.3 LTS`

`torch:2.5.1`

`CUDA： 12.1`

`Python： 3.9`

## 2. 测试方法

### 2.1 测试代码
```bash
python gemm_eval.py --batch_size 40000 --shape_row 40000 --shape_col 40000 --plot_mode True --output shape_40000_int8_info.txt --data_type INT8
```

### 2.2 参数说明

矩阵乘法的shape: $(batch, row) \times (row, col)$
- `batch_size`：batch大小，如果没有指定，则会测试`[32, 64, 128, 256, 512,1024,2048,4096,8192,16384,40000]`这几个大小
- `shape_row`：矩阵的行数
- `shape_col`：矩阵的列数
- `output`：输出文件名, 输出文件都会放在`output`文件夹下
- `data_type`：数据类型，支持`INT8`、`FP16`、`FP32`

在run.bash中提供了几个测试的例子

## 3. 锁频脚本

提供了锁频的脚本，可以通过`lock_freq.bash`来锁频（锁在了4090的boost clock:2520MHZ,可以根据自己的需要修改，同时需要sudo权限）