# Deep Learning Project Template for PyTorch

## Features
+ Tensorboard / TensorboardX / [wandb](https://wandb.ai/site) support
+ AverageMeter / ProgressMeter
+ Early Stopping
+ NVIDIA [DALI](https://developer.nvidia.com/dali) support
+ [einops](https://github.com/arogozhnikov/einops) support
---
## Code Structure
+ ```data```  dir:
    Dataset and dataloader code are implement here.
+ ```model``` dir:
    Implement some of the computer vision model here.
+ ```utils``` dir:
    Some of the useful function implement for the DL training process. (Record especially)
---
## Installation requirements library
```python=
pip install requirements.txt
```
---
## Usage
```python=
import tqdm
import model
from utils import AverageMeter, EarlyStopping, ProgressMeter

import torch
from torch.nn import functional as F

densenet_121 = model.densenet121(pretrained = True)
early_stop = EarlyStopping(
    patience = args.early_stop,
    verbose = True,
    delta = 1e-3
)

for epoch in range(1, epochs + 1):
    # Train model
    train_losses = AverageMeter('Train Loss', ':.4e')
    train_top1 = AverageMeter('Acc@1', ':6.2f')
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

        acc1 = accuracy(output, target)
        train_top1.update(acc1[0].item(), output.size(0))

        scheduler.step(train_losses.avg)
        early_stop(val_loss.avg, model)
        if early_stop.early_stop_flag:
            print(f"Epoch [{epoch} / {epochs}]: early stop")
            break
```
---
# Runnung experiments
```python=
python main.py --epochs 10 --batch-size 64 --early-stop 10
```
---
# Reference
1. **D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri.
Learning spatiotemporal features with 3d convolutional networks.
In 2015 IEEE International Conference on Computer
Vision (ICCV), pages 4489–4497. IEEE, 2015.**
2. **Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, pages 770–778, 2016.**
3. **G. Huang, Z. Liu, K. Q. Weinberger, and L. Maaten. Densely
connected convolutional networks. In CVPR, 2017.**
4. **LeCun, Y., Bottou, L., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE**
5. **O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015**
6. **O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234–241. Springer, 2015**
7. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth
16x16 words: Transformers for image recognition at scale. In: ICLR (2021)**
---
# TODOs
- [ ] More Computer Vision related model
- [ ] Optimizer Wrapper (Ex: LookaHead)
- [ ] Use config to construct model
- [x] tensorboardX logger support
- [x] wandb logger support
---
# Author
**Hong-Jia Chen**

