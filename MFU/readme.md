# MFU Calculate For Deit Model

## Code Implementation Approach
Refer to [feishu link](https://kadxsrk5f5d.feishu.cn/wiki/YSoZwjxMNifjjdkLZzFc1dnQn0e?from=from_copylink)

## Execution

### Environment Setup
```bash
pip install torch torchvision
```

### Command For single GPU
```bash
# Given the FLOPs of the model, output the MFU for each model, unit in MACs  
python main.py --flops 50 --output flops_50G.txt
# Given the params of the model, output the MFU for each model, unit in M
python main.py --params 50 --output params_50M.txt
```

### Command For multiple GPUs
```bash
# given the depth and hidden_dim of the model, output the MFU and throughput for each model
python -m torch.distributed.launch --nproc_per_node=8 mfu_cal_ddp.py --depth 12 --hidden_dim 768 --batch_size 32 --output mfu_deit_base_bs32.txt
```