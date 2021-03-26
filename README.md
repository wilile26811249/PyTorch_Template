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
usage: train.py [-h] [--gpu_devices GPU_DEVICES [GPU_DEVICES ...]]
                [--batch-size N] [--test-batch-size N] [--epochs N] [--lr LR]
                [--warmup-epoch WARMUP_EPOCH] [--gamma M] [--no-cuda]
                [--dry-run] [--seed S] [--early-stop EARLY_STOP]
                [--model-path MODEL_PATH] [--wandb-name WANDB_NAME]
                [--train-data-path TRAIN_DATA_PATH]
                [--test-data-path TEST_DATA_PATH]

PyTorch Medical Project

optional arguments:
  -h, --help            show this help message and exit
  --gpu_devices GPU_DEVICES [GPU_DEVICES ...]
                        Select specific GPU to run the model
  --batch-size N        Input batch size for training (default: 64)
  --test-batch-size N   Input batch size for testing (default: 64)
  --epochs N            Number of epochs to train (default: 20)
  --lr LR               Learning rate (default: 0.01)
  --warmup-epoch WARMUP_EPOCH
                        Warmup epoch (default: 10)
  --gamma M             Learning rate step gamma for StepLR (default: 0.1)
  --no-cuda             Disables CUDA training(default: False)
  --dry-run             Quickly check a single pass
  --seed S              Random seed (default: 1)
  --early-stop EARLY_STOP
                        After n consecutive epochs,val_loss isn't improved
                        then early stop
  --model-path MODEL_PATH
                        For Saving the current Model(default: checkpoint.pt)
  --wandb-name WANDB_NAME
                        Setting run name for wandb (default: "").
  --train-data-path TRAIN_DATA_PATH
                        Path for the training data (default:
                        ./data/retina/train
  --test-data-path TEST_DATA_PATH
                        Path for the testing data (default:
                        ./data/retina/train
```
---
# Running experiments
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

