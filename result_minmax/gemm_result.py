
import os
import argparse

def parse_args():
    # txt path
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path", type=str, default="result_minmax")
    parser.add_argument("--data_type", type=str, default="NotINT8")
    args = parser.parse_args()
    return args

def get_files(txt_path):
    files = os.listdir(txt_path)
    txt_files = []
    for file in files:
        if file.endswith(".txt"):
            txt_files.append(file)
    return txt_files

def main():
    args = parse_args()
    txt_files = get_files(args.txt_path)
    result = {}
    max_MFU = -1
    max_id = None
    min_MFU = 100000000
    min_id = None
    for txt_file in txt_files:
        with open(os.path.join(args.txt_path, txt_file), "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                if 'shape' in line:
                    shape_begin_index = line.find("shape: ")
                    shape = line[shape_begin_index + len("shape: "):-1]
                # 获取batch_size，MFU，THROUGHPUT,FLOPS
                if "batch_size" in line:
                    batch_size_begin_index = line.find("batch_size: ")
                    batch_size_end_index = line.find(" MFU: ")
                    batch_size = line[batch_size_begin_index + len("batch_size: "):batch_size_end_index]
                    MFU_begin_index = line.find("MFU: ")
                    if args.data_type == "INT8":
                        MFU_end_index = line.find(" OPS: ")
                    else:
                        MFU_end_index = line.find(" FLOPS:")
                    MFU = line[MFU_begin_index + len("MFU: "):MFU_end_index]
                    result[int(batch_size)] = float(MFU)
                    if float(MFU) > max_MFU:
                        max_MFU = float(MFU)
                        max_id = str(batch_size) + "x" + shape
                    if float(MFU) < min_MFU:
                        min_MFU = float(MFU)
                        min_id = str(batch_size) + "x" + shape
    print(max_id+":"+ str(max_MFU))
    print(min_id+":"+ str(min_MFU))
    
if __name__ == "__main__":
    main()