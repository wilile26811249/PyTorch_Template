from os import path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
    def __init__(self,
        name: str,
        fmt: Optional[str] = ':f',
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,
        val: float,
        n: Optional[int] = 1
    ) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: Optional[str] = "",
        batch_info: Optional[str] = ""
    ) -> None:
        self.batch_fmster = self._get_batch_fmster(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.batch_info = batch_info

    def display(self, batch):
        self.info = [self.prefix + self.batch_info + self.batch_fmster.format(batch)]
        self.info += [str(meter) for meter in self.meters]
        print('\t'.join(self.info))

    def _get_batch_fmster(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class EarlyStopping(object):
    """
    Arg
    """
    def __init__(self,
        patience: int = 7,
        verbose: Optional[bool] = False,
        delta: Optional[float] = 0.0,
        path: Optional[str] = "checkpoint.pt"
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop_flag = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.verbose = verbose
        self.path = path

    def __call__(self, val_loss, model):
        score = abs(val_loss)
        if self.best_score is None:
            self.best_score = score
            self.save_model(val_loss, model)
        elif val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
                print(f"Best val loss: {self.val_loss_min}  Current val loss: {score}")
            if self.counter >= self.patience:
                self.early_stop_flag = True
        else:
            self.best_score = score
            self.save_model(val_loss, model)
            self.counter = 0

    def save_model(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def accuracy(output, target, topk = (1,)):
    """
    Computes the accuracy over the top k predictions
    """
    with torch.no_grad():
        max_k = max(topk)
        batch_size = output.size(0)

        _, pred = output.topk(max_k,
            dim = 1,
            largest = True,
            sorted = True
        )
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[: k].view(-1).float().sum(0, keepdim = True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def init_weights(m):
    """
    Initiate the parameters either from existing checkpoint or from
    scratch.
    """
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean = 0, std = 1)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
