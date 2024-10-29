import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
import time

# Actually, they are just local rank and local world size
def metric_ddp(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # create default process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # initialize model
    metric = torchmetrics.Accuracy()

    cuda_device = rank

    batch_size = 32
    num_workers = 4

    imagenet_1k_dir = "imagenet_200"
    val_dir = os.path.join(imagenet_1k_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_set = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_sampler = DistributedSampler(dataset=val_set)

    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             sampler=val_sampler,
                                             pin_memory=True)

    model = torchvision.models.resnet18(pretrained=True)
    model.metric = metric
    model = model.to(cuda_device)

    model = DDP(model, device_ids=[rank])

    model.eval()

    criterion = nn.CrossEntropyLoss().cuda(cuda_device)
    time_begin = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if cuda_device is not None:
                images = images.to(cuda_device, non_blocking=True)
            if torch.cuda.is_available():
                target = target.to(cuda_device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            acc = metric(output, target)

            print_freq = 10
            if rank == 0 and i % print_freq == 0:  # print only for rank 0
                print(f"Accuracy on batch {i}: {acc}")
            if i > 10:
                break
        # metric on all batches and all accelerators using custom accumulation
        # accuracy is same across both accelerators
        acc = metric.compute()
        print(f"Accuracy on all data: {acc}, accelerator rank: {rank}")

        # Reseting internal state such that metric ready for new data
        metric.reset()
    time_end = time.time()
    print("Time taken: ", time_end - time_begin)
    # cleanup
    dist.destroy_process_group()


if __name__ == "__main__":

    world_size = 8  # number of gpus to parallize over
    mp.spawn(metric_ddp, args=(world_size, ), nprocs=world_size, join=True)

