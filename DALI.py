import argparse
import os
import shutil
import time

import nvidia.dali.ops as ops
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from caffe2.python.visualize import NCHW, NHWC
from numpy.compat.py3k import long
from numpy.ma.core import squeeze
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import (DALIClassificationIterator,
                                        DALIGenericIterator)
from torch import optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import RandomResizedCrop
from tqdm import tqdm

import model

DOG_CAT_PATH = "/data/dogcat/"
CROP_SIZE = 224

# Training settings
parser = argparse.ArgumentParser(description = 'NVIDIA DALI Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
args = parser.parse_args()

cudnn.benchmark = True


class HybridTrainPipe(Pipeline):
    """
    FileReader:
        to read files from the drive
    HostDecoder:
        to decode images to RGB format
    """
    def __init__(self,
        batch_size,
        num_threads,
        device_id,
        data_dir,
        crop = 224,
        dali_cpu = False
    ):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads,
                                              device_id, seed = 12 + device_id)
        self.input = ops.FileReader(file_root = data_dir, random_shuffle = True)
        # let user decide which pipeline works him
        if dali_cpu:
            dali_device = 'cpu'
            self.decode = ops.HostDecoder(device = dali_device, output_type = types.RGB)
        else:
            dali_device = 'gpu'
            self.decode = ops.ImageDecoder(device = 'mixed',
                                            output_type = types.RGB,
                                            device_memory_padding = 211025920,
                                            host_memory_padding = 140544512
            )

        self.rrc = ops.RandomResizedCrop(device = dali_device, size = (crop, crop))
        self.cmnp = ops.CropMirrorNormalize(device = 'gpu',
                                            output_type = types.FLOAT,
                                            output_layout = types.NCHW,
                                            crop = (crop, crop),
                                            image_type = types.RGB,
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        )
        self.coin = ops.CoinFlip(probability = 0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name = "Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images.gpu(), mirror = rng)
        return [output, self.labels]


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    elapsed_time = 0.0
    end = time.time()

    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()

        data_var = Variable(data)
        target_var = Variable(target)

        optimizer.zero_grad()
        output = model(data_var)
        loss = criterion(output, target_var)

        # Compute gradient and do optimizer step
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        elapsed_time = elapsed_time + time.time() - end
        end = time.time()

        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print(f"Average Time(batch time): {elapsed_time / args.log_interval}")
            elapsed_time = 0.0
            if args.dry_run:
                break


def main():
    global args, DOG_CAT_PATH, CROP_SIZE
    args.world_size = 1

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    net = model.resnet50(num_classes = 2).to(device)

    # Define loss function and optimizer
    criterion = nn.functional.cross_entropy
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    # Define Training Arguments
    train_kwargs = {'batch_size' : args.batch_size}
    if use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)

    # DALI Loader
    pipe = HybridTrainPipe(batch_size = args.batch_size,
                           num_threads = 4,
                           device_id = 7,
                           data_dir = DOG_CAT_PATH
    )
    pipe.build()

    train_loader = DALIClassificationIterator(pipe, size = int(pipe.epoch_size("Reader") / args.world_size))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    for epoch in range(1, args.epochs + 1):
        train(args, net, device, train_loader, criterion, optimizer, epoch)
        scheduler.step(1)

    if args.save_model:
        torch.save(net.state_dict(), "DALI_final.pt")


if __name__ == "__main__":
    main()