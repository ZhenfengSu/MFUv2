import matplotlib.pyplot as plt  
import argparse  

def get_output_info(output_path):  
    BATCH_SIZE_list = []  
    MFU_list = []   
    FLOPS_model_list = []  
    with open(output_path, 'r') as f:  
        all_lines = f.readlines()  
        for line in all_lines:  
            if 'batch_size' in line:  
                batch_begin_index = line.index('batch_size: ')  
                batch_end_index = line.index(' MFU')  
                batch_size = int(line[batch_begin_index + 12:batch_end_index])  
                MFU_begin_index = line.index('MFU: ')  
                MFU_end_index = line.index(' FLOPS:')  
                MFU = float(line[MFU_begin_index + 5:MFU_end_index])  
                FLOPS_model_begin_index = line.index('FLOPS: ')  
                FLOPS_model_end_index = line.index('\n')  
                FLOPS_model = float(line[FLOPS_model_begin_index + 7:FLOPS_model_end_index])  
                BATCH_SIZE_list.append(batch_size)  
                MFU_list.append(MFU)   
                FLOPS_model_list.append(FLOPS_model)  
            if 'shape' in line:  
                model_type_index = line.index('shape: ')  
                model_type = line[model_type_index + len('shape: '):-1]  
    return BATCH_SIZE_list, MFU_list, FLOPS_model_list, model_type  

def plot_MFU_THROUGHPUT_FLOPS(output_path, save_path):  
    batch_size_list, MFU_list, FLOPS_model_list, model_type = get_output_info(output_path)  
    
    # Setup figure size  
    plt.figure(figsize=(10, 12))  
    
    # Plot MFU  
    plt.subplot(2, 1, 1)  
    plt.plot(batch_size_list, [m * 100 for m in MFU_list], 'ro-', label='MFU')  # Convert to percentage  
    plt.xlabel('Batch Size', fontsize=12)  
    plt.ylabel('MFU (%)', fontsize=12)  # Add percentage unit  
    plt.title('MFU', fontsize=14)  
    plt.xticks(batch_size_list, rotation=45)  
    plt.grid(True)  
    plt.legend()  
    
    # Plot FLOPS Model  
    plt.subplot(2, 1, 2)  
    plt.plot(batch_size_list, [f for f in FLOPS_model_list], 'bo-', label='FLOPS Model')  # Convert to TFLOPS  
    plt.xlabel('Batch Size', fontsize=12)  
    plt.ylabel('FLOPS Model (TFLOPS)', fontsize=12)  # Add TFLOPS unit  
    plt.title('FLOPS Model', fontsize=14)  
    plt.xticks(batch_size_list, rotation=45)  
    plt.grid(True)  
    plt.legend()  

    # Adjust layout to prevent overlap  
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    
    # Plot Supertitle  
    name = args.save_path.split('.')[0]
    plt.suptitle('MFU THROUGHOUT FLOPS of ' + str(name).upper(), fontsize=16)  
    
    # Save the figure  
    plt.savefig(save_path)  

def get_args_parser():  
    parser = argparse.ArgumentParser(description="PyTorch Distributed DataParallel")  
    parser.add_argument("--output-path", type=str, default='output_4096.txt', help='output_path')  
    parser.add_argument("--save-path", type=str, default='output_4096.png', help='save_path')  
    args = parser.parse_args()  
    return args  

if __name__ == '__main__':  
    args = get_args_parser()  
    plot_MFU_THROUGHPUT_FLOPS(args.output_path, args.save_path)  
    print("done")