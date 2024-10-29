import os
image_path = '/root/rank_project/MFUv2/DDP_EVAL/tiny-imagenet-200/val/images'
annotation_path = '/root/rank_project/MFUv2/DDP_EVAL/tiny-imagenet-200/val/val_annotations.txt'

if not os.path.exists('imagenet_200'):
    os.makedirs('imagenet_200')
if not os.path.exists('imagenet_200/val'):
    os.makedirs('imagenet_200/val')

with open(annotation_path, 'r') as f:
    all_lines = f.readlines()
    for line in all_lines:
        line = line.split("	")
        image_name = line[0]
        image_class = line[1]
        if not os.path.exists('imagenet_200/val/' + image_class):
            os.makedirs('imagenet_200/val/' + image_class)
        os.system('cp ' + image_path + '/' + image_name + ' imagenet_200/val/' + image_class + '/' + image_name)
print("done")