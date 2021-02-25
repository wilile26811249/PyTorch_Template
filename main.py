import argparse
import time

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

import model
from data import get_dataloaders
from data.transformation import train_transform, val_transform
from utils import AverageMeter, EarlyStopping, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='Number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                    help='Learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training(default: False)')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='Quickly check a single pass')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='Random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')
parser.add_argument('--early-stop', type=int, default=10,
                    help="After n consecutive epochs,val_loss isn't improved then early stop")
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model(default: False)')


#===============MyDataset========================
def train_MyDataset(args, model, device, optimizer, train_loader, val_loader):
    early_stop = EarlyStopping(
        patience = args.early_stop,
        verbose = True,
        delta = 1e-3
    )

    for epoch in range(1, args.epochs + 1):
        # Train model
        train_losses = AverageMeter('Train Loss', ':.4e')
        model.train()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
        for _, data_dict in tqdm(enumerate(train_loader)):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            train_losses.update(loss.item(), data.size(0))
            loss.backward()
            optimizer.step()
        scheduler.step(1)

        # Validate model
        batch_time = AverageMeter('Time', ':6.3f')
        val_loss = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        progress = ProgressMeter(
            num_batches = len(val_loader),
            meters = [top1, val_loss, batch_time],
            prefix = "Epoch: [{}]".format(epoch),
            batch_info = "  Iter: "
        )

        model.eval()
        end = time.time()
        for it, data_dict in enumerate(val_loader):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)

            # Compute output and loss
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss

            # Measure accuracy,  update top1@ and losses
            acc1 = accuracy(output, target)
            val_loss.update(loss, data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            progress.display(it)

            # Measure the elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        early_stop(val_loss.avg, model)
        if early_stop.early_stop_flag:
            print(f"Epoch [{epoch} / {args.epochs}]: early stop")
            break
    model.load_state_dict(torch.load(early_stop.path))
    return model, train_losses.avg, early_stop.best_score


def test_MyDataset(model, device, test_loader):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    with torch.no_grad():
        for _, data_dict in enumerate((test_loader)):
            data, target = data_dict['image'].to(device), data_dict['targets'].to(device)

            # Compute output and loss
            output = model(data)

            # Measure accuracy,  update top1@
            test_acc = accuracy(output, target)
            top1.update(test_acc[0].item(), data.size(0))
    print(f"Test accuracy: {top1.avg}")

def main():
    # Training settings
    args = parser.parse_args()
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.use_cuda:
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    train_kwargs = {'batch_size' : args.batch_size, 'shuffle' : True}
    test_kwargs = {'batch_size' : args.test_batch_size, 'shuffle' : True}
    if args.use_cuda:
        cuda_kwargs = {
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    path = './data/test1'
    train_dl, val_dl, test_dl = get_dataloaders(
        train_dir = path,
        test_dir = path,
        train_transform = train_transform,
        test_transform = val_transform,
        **train_kwargs
    )

    net = model.resnet18(num_classes = 2).to(device)
    optimizer = optim.SGD(net.parameters(), lr = args.lr)

    net, train_loss, val_loss = train_MyDataset(args, net, device, optimizer, train_dl, val_dl)
    print(f"Final  --->  Train loss: {train_loss}  Val loss: {val_loss}")

    test_MyDataset(net, device, test_dl)
    if args.save_model:
        torch.save(net.state_dict(), "latest_model.pt")


if __name__ == '__main__':
    main()
