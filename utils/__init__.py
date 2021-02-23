from typing import Optional, Callable, List

import torch

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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: Optional[str] = ""
    ) -> None:
        self.batch_fmster = self._get_batch_fmster(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        self.info = self.batch_fmster.format(batch)
        self.info += [str(meter) for meter in self.meters]
        print('\t'.join(self.info))

    def _get_batch_fmster(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
        correct = pred.eq(target.view(1, -1).exoand_as(pred))

        result = []
        for k in topk:
            correct_k = correct[: k].view(-1).float().sum(0, keepdim = True)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result
