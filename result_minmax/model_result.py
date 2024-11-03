import argparse

def parse_args():
    # txt path
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path", type=str, default="result_minmax/result.txt")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    with open(args.txt_path, "r") as f:
        all_lines = f.readlines()
        result = {}
        for line in all_lines:
            # 获取batch_size，MFU，THROUGHPUT,FLOPS
            if "batch_size" in line:
                batch_size_begin_index = line.find("batch_size: ")
                batch_size_end_index = line.find(" MFU: ")
                batch_size = line[batch_size_begin_index + len("batch_size: "):batch_size_end_index]
                print("batch_size: ", batch_size)
                MFU_begin_index = line.find("MFU: ")
                MFU_end_index = line.find(" THROUGHPUT: ")
                MFU = line[MFU_begin_index + len("MFU: "):MFU_end_index]
                print("MFU: ", MFU)
                THROUGHPUT_begin_index = line.find("THROUGHPUT: ")
                THROUGHPUT_end_index = line.find(" FLOPS: ")
                THROUGHPUT = line[THROUGHPUT_begin_index + len("THROUGHPUT: "):THROUGHPUT_end_index]
                print("THROUGHPUT: ", THROUGHPUT)
                FLOPS_begin_index = line.find("FLOPS: ")
                FLOPS_end_index = line.find(" LATENCY: ")
                FLOPS = line[FLOPS_begin_index + len("FLOPS: "):FLOPS_end_index]
                print("FLOPS: ", FLOPS)
                result[int(batch_size)] = float(MFU)
        print(result)
        print("max MFU: ", max(result.values()))
        print("max batch_size: ", max(result, key=result.get))
        print("min MFU: ", min(result.values()))
        print("min batch_size: ", min(result, key=result.get))

if __name__ == "__main__":
    main()
                
        
    
