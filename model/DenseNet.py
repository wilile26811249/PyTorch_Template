from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor, optim
from torch.cuda import device
from torch.functional import norm

from .utils import load_state_dict_from_url

