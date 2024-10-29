# 预先设定好的hidden_dim和heads patch_size的选择
hidden_dim_choices_list = [192, 384, 512, 768, 1024, 1280, 1664, 1792, 1920, 2048, 2240, 2432, 2560, 2816, 3072, 3328, 3712, 3840, 3968, 4096]

hidden_dim_choices = {
    '192':[ 16 , 3], # patch_size, heads
    '384':[ 16 , 6],
    '512':[ 16 , 8],
    '768':[ 16 , 12],
    '1024':[ 16 , 16],
    '1280':[ 14 , 16],
    '1664':[ 14 , 16],
    '1792':[ 14 , 16],
    '1920':[ 14 , 16],
    '2048':[ 14 , 16],
    '2240':[ 14 , 16],
    '2432':[ 14 , 19],
    '2560':[ 14 , 20],
    '2816':[ 14 , 22],
    '3072':[ 14 , 24],
    '3328':[ 14 , 26],
    '3712':[ 14 , 29],
    '3840':[ 14 , 30],
    '3968':[ 14 , 31],
    '4096':[ 14 , 32],
 }
'''计算flops的函数'''
# deit flops directly

def flops_transformer_block(hidden_dim, target_seq_len):
    s = target_seq_len
    h = hidden_dim
    return 12 * h * h * s + 2 * h * s * s

def flops_patch_embedding(hidden_dim, patch_size):
    h = hidden_dim
    p = patch_size
    return (3*h*p*p + h) * 224*224 / (p*p)

def flops_head(hidden_dim, num_classes):
    h = hidden_dim
    c = num_classes
    return h * c

def flops_deit(hidden_dim,patch_size,num_classes,depth):
    target_seq_len = (224 // patch_size) ** 2 + 1 # cls token + 1
    return (flops_patch_embedding(hidden_dim, patch_size) +
            sum(flops_transformer_block(hidden_dim, target_seq_len) for _ in range(depth)) +
            flops_head(hidden_dim, num_classes))
'''计算params的函数'''
# deit params directly
def params_transformer_block(hidden_dim):
    # attention 
    qkv_params = 3 * hidden_dim * hidden_dim + 3 * hidden_dim
    proj_params = hidden_dim * hidden_dim + hidden_dim
    mpl_fc1_params = 4 * hidden_dim * hidden_dim + 4*hidden_dim
    mpl_fc2_params = hidden_dim * 4 * hidden_dim + hidden_dim
    return qkv_params + proj_params + mpl_fc1_params + mpl_fc2_params

def params_patch_embedding(hidden_dim, patch_size):
    h = hidden_dim 
    p = patch_size
    return 3 * h * p * p + h

def params_head(hidden_dim, num_classes):
    h = hidden_dim
    c = num_classes
    return h * c + c

def cls_token_params(hidden_dim):
    return hidden_dim

def positional_embedding_params(hidden_dim, patch_size):
    h = hidden_dim
    p = patch_size
    
    return 1 * h * (224 // p) ** 2

def params_deit(hidden_dim, patch_size, num_classes, depth):
    print("params_deit")
    print("patch_embedding")
    print(params_patch_embedding(hidden_dim, patch_size)/1e6)
    print("transformer_block")
    print(params_transformer_block(hidden_dim)/1e6)
    print("head")
    print(params_head(hidden_dim, num_classes)/1e6)
    # 与flops_count部分的代码一致,不去计算cls_token_params和positional_embedding_params
    return params_patch_embedding(hidden_dim, patch_size) + \
            sum(params_transformer_block(hidden_dim) for _ in range(depth)) + \
            params_head(hidden_dim, num_classes)
'''二分查找,查找给定flops下的最大的hidden_dim'''
# 二分查找的状态判断
def binary_search_hidden_dim_state(hidden_dim,flops):
    # 先计算去除head和patch_embedding的flops
    class_num = 1000
    hidden_dim = hidden_dim
    hidden_dim_string = str(hidden_dim)
    patch_size = hidden_dim_choices[hidden_dim_string][0]
    deit_head_flops = flops_head(hidden_dim, class_num)
    deit_patch_embed_flops = flops_patch_embedding(hidden_dim, patch_size)
    deit_block_flops_total = flops - deit_head_flops - deit_patch_embed_flops
    # 计算transformer_block的flops
    transformer_block_flops = flops_transformer_block(hidden_dim, (224 // patch_size) ** 2 + 1)
    if transformer_block_flops > deit_block_flops_total:
        return 1
    else:
        return 0
    
# 二分查找的函数
def binary_search_hidden_dim(flops):
    left = 0
    right = len(hidden_dim_choices_list) - 1
    while left < right:
        mid = (left + right) // 2
        hidden_dim = hidden_dim_choices_list[mid]
        state = binary_search_hidden_dim_state(hidden_dim, flops)
        if state == 1:
            right = mid
        else:
            left = mid + 1
    return hidden_dim_choices_list[left]

'''二分查找,查找给定params下的最大的hidden_dim'''   
# 二分查找的状态判断
def binary_search_hidden_dim_state_params(hidden_dim,params):
    # 先计算去除head和patch_embedding的params
    class_num = 1000
    hidden_dim = hidden_dim
    hidden_dim_string = str(hidden_dim)
    patch_size = hidden_dim_choices[hidden_dim_string][0]
    deit_head_params = params_head(hidden_dim, class_num)
    deit_patch_embed_params = params_patch_embedding(hidden_dim, patch_size)
    deit_block_params_total = params - deit_head_params - deit_patch_embed_params
    # 计算transformer_block的params
    transformer_block_params = params_transformer_block(hidden_dim)
    if transformer_block_params > deit_block_params_total:
        return 1
    else:
        return 0

# 二分查找的函数
def binary_search_hidden_dim_params(params):
    left = 0
    right = len(hidden_dim_choices_list) - 1
    while left < right:
        mid = (left + right) // 2
        hidden_dim = hidden_dim_choices_list[mid]
        state = binary_search_hidden_dim_state_params(hidden_dim, params)
        if state == 1:
            right = mid
        else:
            left = mid + 1
    return hidden_dim_choices_list[left]    

'''给定hidden_dim和flops或者params，返回对应的depth'''
# 给定flops
def get_depth_flops(hidden_dim, flops):
    # 先计算去除head和patch_embedding的flops
    class_num = 1000
    hidden_dim = hidden_dim
    hidden_dim_string = str(hidden_dim)
    patch_size = hidden_dim_choices[hidden_dim_string][0]
    deit_head_flops = flops_head(hidden_dim, class_num)
    deit_patch_embed_flops = flops_patch_embedding(hidden_dim, patch_size)
    deit_block_flops_total = flops - deit_head_flops - deit_patch_embed_flops
    # 计算transformer_block的flops
    transformer_block_flops = flops_transformer_block(hidden_dim, (224 // patch_size) ** 2 + 1)
    # 计算depth
    depth = deit_block_flops_total // transformer_block_flops
    return depth

# 给定params
def get_depth_params(hidden_dim, params):
    # 先计算去除head和patch_embedding的params
    class_num = 1000
    hidden_dim = hidden_dim
    hidden_dim_string = str(hidden_dim)
    patch_size = hidden_dim_choices[hidden_dim_string][0]
    deit_head_params = params_head(hidden_dim, class_num)
    deit_patch_embed_params = params_patch_embedding(hidden_dim, patch_size)
    deit_block_params_total = params - deit_head_params - deit_patch_embed_params
    # 计算transformer_block的params
    transformer_block_params = params_transformer_block(hidden_dim)
    # 计算depth
    depth = deit_block_params_total // transformer_block_params
    return depth
    

# 1.给定 hidden_dim和depth，返回deit模型信息
# 2.给定flops,返回不同depth和hidden_dim的deit模型信息
# 3.给定params,返回不同depth和hidden_dim的deit模型信息
def get_deit_model_info_map(hidden_dim, depth, flops, params):
    # 首先判断不能三个参数都给定
    model_info = 0
    flag = None 
    if hidden_dim is not None and depth is not None:
        model_info += 1
        flag = 1
    if flops is not None:
        model_info += 1
        flag = 2
    if params is not None:
        model_info += 1
        flag = 3
    assert model_info == 1, "hidden_dim, depth, flops, params不能同时给定"
    if flag == 1:
        hidden_dim_string = str(hidden_dim)
        patch_size, heads = hidden_dim_choices[hidden_dim_string]
        num_classes = 1000
        model_info_map = {
            "hidden_dim": hidden_dim,
            "patch_size": patch_size,
            "heads": heads,
            "num_classes": num_classes,
            "depth": depth
        }
        return [model_info_map]
    elif flag == 2:
        # 给定flops,返回不同depth和hidden_dim的deit模型
        # 首先计算出最大的hidden_dim
        max_hidden_dim = binary_search_hidden_dim(flops)
        # 保存所有的hidden_dim和depth
        model_info_list = []
        index_hidden_dim = hidden_dim_choices_list.index(max_hidden_dim)
        for i in range(index_hidden_dim + 1):
            depth = get_depth_flops(hidden_dim_choices_list[i], flops)
            hidden_dim = hidden_dim_choices_list[i]
            hidden_dim_string = str(hidden_dim)
            patch_size, heads = hidden_dim_choices[hidden_dim_string]
            num_classes = 1000
            model_info_map = {
                "hidden_dim": hidden_dim_choices_list[i],
                "patch_size": patch_size,
                "heads": heads,
                "num_classes": num_classes,
                "depth": int(depth)
            }
            model_info_list.append(model_info_map)
        return model_info_list
    elif flag == 3:
        # 给定params,返回不同depth和hidden_dim的deit模型
        # 首先计算出最大的hidden_dim
        max_hidden_dim = binary_search_hidden_dim_params(params)
        # 保存所有的hidden_dim和depth
        model_info_list = []
        index_hidden_dim = hidden_dim_choices_list.index(max_hidden_dim)
        for i in range(index_hidden_dim + 1):
            depth = get_depth_params(hidden_dim_choices_list[i], params)
            hidden_dim_string = str(hidden_dim)
            patch_size, heads = hidden_dim_choices[hidden_dim_string]
            num_classes = 1000
            model_info_map = {
                "hidden_dim": hidden_dim_choices_list[i],
                "patch_size": patch_size,
                "heads": heads,
                "num_classes": num_classes,
                "depth": int(depth)
            }
            model_info_list.append(model_info_map)
        return model_info_list
               

if __name__ == '__main__':
    # deit-base
    hidden_dim = 768
    patch_size = 16
    num_classes = 1000
    depth = 12
    heads = 12
    from deit_entiremodel import vit_models, Layer_scale_init_Block
    from deit_entiremodel import deit_base_patch16_LS
    model = deit_base_patch16_LS()
    model_info_map = {
        "hidden_dim": hidden_dim,
        "patch_size": patch_size,
        "heads": heads,
        "num_classes": num_classes,
        "depth": depth
    }
    from mfu_cal import mfu, mfu_throughput
    MFU = mfu(model, 24, model_info_map)
    print("the deit-base MFU is: " + str(MFU*100) + '%')
    MFU, throughput = mfu_throughput(model, 24, model_info_map)
    print("the deit-base throughput is: " + str(throughput) + 'images/s')
    print("the deit-base MFU is: " + str(MFU*100) + '%')
    flops = flops_deit(hidden_dim, patch_size, num_classes, depth)
    print(f'FLOPs: {flops / 1e9:.3f}G Macs')
    params = params_deit(hidden_dim, patch_size, num_classes, depth)
    print(f'Params: {params / 1e6:.3f}M' )
    